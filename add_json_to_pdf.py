# Mock add_json_to_pdf.py - PDF annotation utility
import json
from pathlib import Path

# -*- coding: utf-8 -*-
"""
Created on Fri Aug  1 02:45:06 2025

@author: hellpanderrr
"""

import pymupdf as fitz  # PyMuPDF
import textwrap

VALUE_WRAP_WIDTH = 100


def wrap_text_preserving_newlines(text: str, width_chars: int) -> list[str]:
    """
    Wraps text to a specified character width while preserving existing newline characters.

    Args:
        text: The input string, which may contain '\\n'.
        width_chars: The maximum width of any line in characters.

    Returns:
        A list of strings, ready to be drawn line-by-line.
    """
    if not isinstance(text, str):
        text = str(text)

    output_lines = []
    # 1. Split the text into its original lines based on the \n character
    original_lines = (text or "").split('\n')

    for line in original_lines:
        # 2. Wrap each original line individually
        wrapped_sublines = textwrap.wrap(
            line,
            width=width_chars,
            replace_whitespace=False,
            break_long_words=True,
            # drop_whitespace helps clean up lines that might be just spaces
            drop_whitespace=True)

        # If a line was empty (from a "\n\n"), textwrap returns [].
        # We must add an empty string to preserve the blank line.
        if not wrapped_sublines:
            output_lines.append("")
        else:
            # 3. Add the resulting (potentially multi-line) parts to our final list
            output_lines.extend(wrapped_sublines)

    return output_lines


def add_json_to_pdf(input_pdf: str,
                    json_results: dict[str, list[dict]],
                    output_pdf: str,
                    one_based_idx=False):
    """
    Adds tabular representations of LLM JSON results to the right side of a PDF's pages.

    Args:
        input_pdf: Path to the original PDF file.
        json_results: A dictionary where keys are version names (e.g., "LLM_v1") 
                      and values are the list of JSON objects returned by the LLM.
        output_pdf: Path to save the augmented PDF.
    """
    doc = fitz.open(input_pdf)

    # --- Constants for layout ---
    FONTSIZE_TITLE = 12
    FONTSIZE_HEADER = 10
    FONTSIZE_BODY = 9
    LINE_HEIGHT = 15
    ROW_PADDING = 5
    MIN_ROW_HEIGHT = 20

    # --- 1. Pre-calculation Step ---
    col_widths = [150, 150, 400]
    table_width = sum(col_widths) + 40

    max_required_width = 0
    page_heights = [page.rect.height for page in doc]

    results_by_page = {}
    from copy import copy
    json_results = copy(json_results)
    for version, results_list in json_results.items():
        for item in results_list:
            page_num = item.get("page_number")
            if one_based_idx:
                page_num -= 1
            #if page_num > 9:
            #    page_num -= 1
            if page_num is not None:
                if page_num not in results_by_page:
                    results_by_page[page_num] = {}
                results_by_page[page_num][version] = item
                #print(page_num, results_by_page[page_num][version]['page_number'])
                #if page_num > 9:
                #    results_by_page[page_num][version]['page_number'] -=1

    for i, page in enumerate(doc):
        original_width = page.rect.width

        page_num_key = i
        page_predictions = results_by_page.get(page_num_key, {})

        required_width = original_width + (table_width * len(page_predictions))
        if required_width > max_required_width:
            max_required_width = required_width

        max_table_height = 0
        if page_predictions:
            for version, data in page_predictions.items():
                current_height = 50
                flat_data = []
                flat_data.append(("spec_id", "", data.get("spec_id")))
                flat_data.append(
                    ("product_type", "", data.get("product_type")))
                flat_data.append(
                    ("product_category", "", data.get("product_category")))
                flat_data.append(("notes", "", data.get("notes")))
                flat_data.append(
                    ("instructions", "", data.get("instructions")))
                flat_data.append(("metadata", "", data.get("metadata")))

                for j, supplier in enumerate(data.get("suppliers", [])):
                    flat_data.append(
                        (f"supplier_{j+1}", "name", supplier.get("name")))
                    flat_data.append(
                        (f"supplier_{j+1}", "type", supplier.get("type")))
                    flat_data.append((f"supplier_{j+1}", "contact",
                                      supplier.get("contact_person")))
                    flat_data.append(
                        (f"supplier_{j+1}", "phone", supplier.get("phone")))
                    flat_data.append(
                        (f"supplier_{j+1}", "email", supplier.get("email")))
                    flat_data.append((f"supplier_{j+1}", "website",
                                      supplier.get("website")))
                    flat_data.append((f"supplier_{j+1}", "address",
                                      supplier.get("address")))

                if isinstance(data.get("attributes"), list):
                    for item in data.get("attributes"):
                        if isinstance(item, dict):
                            flat_data.append(
                                ("attributes", item['k'], item['v']))
                else:
                    for bucket, items in data.get("attributes", {}).items():
                        if isinstance(items, dict):
                            for key, val in items.items():
                                flat_data.append((bucket, key, val))
                        else:
                            flat_data.append((bucket, "", items))

                for field, sub, val in flat_data:
                    wrapped_val = wrap_text_preserving_newlines(val, 100)
                    row_content_height = len(wrapped_val) * LINE_HEIGHT
                    current_height += max(
                        MIN_ROW_HEIGHT, row_content_height + (2 * ROW_PADDING))

                if current_height > max_table_height:
                    max_table_height = current_height

        page_heights[i] = max(page_heights[i], max_table_height + 200)

    max_height = max(page_heights) if page_heights else 1000
    max_height += 200
    # --- 2. Drawing Step ---
    new_doc = fitz.open()

    for page_idx, old_page in enumerate(doc):
        original_width = old_page.rect.width
        original_height = old_page.rect.height

        # Define the dimensions for the new, combined page
        new_page_rect = fitz.Rect(0, 0, max_required_width,
                                  page_heights[page_idx])

        # Create a new blank page with these dimensions
        new_page = new_doc.new_page(width=new_page_rect.width,
                                    height=new_page_rect.height)

        # ### START OF THE ONLY FIX ###
        # Define the target rectangle for the original content. It should be positioned
        # at the top-left corner (0,0) and have the original page's dimensions.
        target_rect = fitz.Rect(0, 0, original_width, original_height)

        # Stamp the original page content into this specific, non-stretched rectangle.
        try:
            new_page.show_pdf_page(target_rect, doc, page_idx)
        except ValueError as e:
            print(e)
        # ### END OF THE ONLY FIX ###

        # Now, `new_page` contains the original content, top-aligned.
        # We can proceed to draw the tables on it.

        x_start = original_width + 20
        page_predictions = results_by_page.get(page_idx, {})

        if not page_predictions:
            continue

        for version_idx, (version,
                          data) in enumerate(page_predictions.items()):
            y_position = 50
            x_position = x_start + (table_width * version_idx)

            # (The entire drawing logic for the tables remains identical)
            new_page.insert_text((x_position, y_position),
                                 version,
                                 fontsize=FONTSIZE_TITLE,
                                 fontname="Helv")
            y_position += 30

            header_rect = fitz.Rect(x_position, y_position,
                                    x_position + sum(col_widths),
                                    y_position + 20)
            new_page.draw_rect(header_rect,
                               color=(0.8, 0.8, 0.8),
                               fill=(0.8, 0.8, 0.8))

            current_x_head = x_position
            for header, width in zip(["Field", "Sub-Field", "Value"],
                                     col_widths):
                new_page.insert_text((current_x_head + 5, y_position + 15),
                                     header,
                                     fontsize=FONTSIZE_HEADER,
                                     fontname="Helv")
                current_x_head += width
            y_position += 20

            rows_to_draw = []
            rows_to_draw.append(("spec_id", "", data.get("spec_id")))
            rows_to_draw.append(("page_number", "", data.get("page_number")))
            rows_to_draw.append(("page_type", "", data.get("page_type")))

            rows_to_draw.append(("product_type", "", data.get("product_type")))
            rows_to_draw.append(
                ("product_category", "", data.get("product_category")))

            for k, supplier in enumerate(data.get("suppliers", [])):
                rows_to_draw.append(
                    (f"supplier_{k+1}", "name", supplier.get("name")))
                rows_to_draw.append(
                    (f"supplier_{k+1}", "type", supplier.get("type")))
                rows_to_draw.append((f"supplier_{k+1}", "contact",
                                     supplier.get("contact_person")))
                rows_to_draw.append(
                    (f"supplier_{k+1}", "phone", supplier.get("phone")))
                rows_to_draw.append(
                    (f"supplier_{k+1}", "email", supplier.get("email")))
                rows_to_draw.append(
                    (f"supplier_{k+1}", "website", supplier.get("website")))
                rows_to_draw.append(
                    (f"supplier_{k+1}", "address", supplier.get("address")))

            rows_to_draw.append(("notes", "", data.get("notes")))
            rows_to_draw.append(("instructions", "", data.get("instructions")))
            rows_to_draw.append(("metadata", "", data.get("metadata")))

            attributes = data.get("attributes", {})
            if isinstance(attributes, list):
                for item in attributes:
                    if isinstance(item, dict):
                        rows_to_draw.append(
                            ("attributes", item['k'], item['v']))

            else:
                for bucket, items in attributes.items():
                    if isinstance(items, dict):
                        for key, val in items.items():
                            rows_to_draw.append((bucket, key, val))
                    else:
                        rows_to_draw.append((bucket, "", items))

            for field, sub_field, value in rows_to_draw:
                wrapped_value = wrap_text_preserving_newlines(
                    value, VALUE_WRAP_WIDTH)

                num_lines = len(wrapped_value)
                row_content_height = num_lines * LINE_HEIGHT
                row_height = max(MIN_ROW_HEIGHT,
                                 row_content_height + (2 * ROW_PADDING))

                row_rect = fitz.Rect(x_position, y_position,
                                     x_position + sum(col_widths),
                                     y_position + row_height)
                new_page.draw_rect(row_rect, color=(0.1, 0.1, 0.1), width=0.5)

                text_start_y = y_position + ROW_PADDING + FONTSIZE_BODY

                new_page.insert_text((x_position + 5, text_start_y),
                                     str(field),
                                     fontsize=FONTSIZE_BODY,
                                     fontname="Helv")
                new_page.insert_text(
                    (x_position + col_widths[0] + 5, text_start_y),
                    str(sub_field),
                    fontsize=FONTSIZE_BODY)

                text_y = text_start_y
                for line in wrapped_value:
                    new_page.insert_text(
                        (x_position + col_widths[0] + col_widths[1] + 5,
                         text_y),
                        line,
                        fontsize=FONTSIZE_BODY,
                        fontfile=r"NotoSans-Regular.ttf",
                        fontname='asdfsfdsf')
                    text_y += LINE_HEIGHT

                y_position += row_height

    new_doc.rewrite_images(
        dpi_threshold=100,  # only process images above 100 DPI
        dpi_target=72,  # downsample to 72 DPI
        quality=60,  # JPEG quality level
        lossy=True,  # include / exclude lossy images
        lossless=True,  # include / exclude lossless images
        bitonal=True,  # include / exclude monochrome images
        color=True,  # include / exclude colored images
        gray=True,  # include / exclude gray-scale images
        set_to_gray=True,  # convert to gray-scale before conversion
    )

    new_doc.ez_save(output_pdf)
    new_doc.close()
    doc.close()  # Close the original document
    print(f"Successfully created augmented PDF at: {output_pdf}")
