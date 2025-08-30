# -*- coding: utf-8 -*-
"""
Created on Wed Jul 30 02:40:58 2025

@author: hellpanderrr
"""
from pymupdf import (
    TEXT_INHIBIT_SPACES,
    TEXT_PRESERVE_LIGATURES,
    TEXT_PRESERVE_WHITESPACE,
)
import tempfile
import os

import pymupdf as fitz
import bisect


def page_layout(page, textout, GRID, fontsize, noformfeed, skip_empty, flags,
                blocks):
    left = page.rect.width  # left most used coordinate
    right = 0  # rightmost coordinate
    rowheight = page.rect.height  # smallest row height in use
    chars = []  # all chars here
    rows = set()  # bottom coordinates of lines
    if noformfeed:
        eop = b"\n"
    else:
        eop = bytes([12])

    # --------------------------------------------------------------------
    def curate_rows(rows, GRID):
        """Make list of integer y-coordinates of lines on page.

        Coordinates will be ascending and differ by 'GRID' points or more."""
        rows = list(rows)
        rows.sort()  # sort ascending
        nrows = [rows[0]]
        for h in rows[1:]:
            if h >= nrows[-1] + GRID:  # only keep significant differences
                nrows.append(h)
        return nrows  # curated list of line bottom coordinates

    # --------------------------------------------------------------------
    def find_line_index(values: list[int], value: int) -> int:
        """Find the right row coordinate (using bisect std package).

        Args:
            values: (list) y-coordinates of rows.
            value: (int) lookup for this value (y-origin of char).
        Returns:
            y-ccordinate of appropriate line for value.
        """
        i = bisect.bisect_right(values, value)
        if i:
            return values[i - 1]
        raise RuntimeError("Line for %g not found in %s" % (value, values))

    # --------------------------------------------------------------------
    def make_lines(chars):
        lines = {}  # key: y1-ccordinate, value: char list
        for c in chars:
            ch, ox, oy, cwidth = c
            y = find_line_index(rows, oy)  # index of origin.y
            lchars = lines.get(y, [])  # read line chars so far
            lchars.append(c)
            lines[y] = lchars  # write back to line

        # ensure line coordinates are ascending
        keys = list(lines.keys())
        keys.sort()
        return lines, keys

    # --------------------------------------------------------------------
    def compute_slots(keys, lines, right, left):
        """Compute "char resolution" for the page.

        The char width corresponding to 1 text char position on output - call
        it 'slot'.
        For each line, compute median of its char widths. The minimum across
        all "relevant" lines is our 'slot'.
        The minimum char width of each line is used to determine if spaces must
        be inserted in between two characters.
        """
        slot = used_width = right - left
        lineslots = {}
        for k in keys:
            lchars = lines[k]  # char tuples of line
            ccount = len(lchars)  # how many
            if ccount < 2:  # if short line, just put in something
                lineslots[k] = (1, 1, 1)
                continue
            widths = [c[3] for c in lchars]  # list of all char widths
            widths.sort()
            line_width = sum(widths)  # total width used by line
            i = int(ccount / 2 + 0.5)  # index of median
            median = widths[i]  # take the median value
            if (line_width / used_width >= 0.3
                    and median < slot):  # if line is significant
                slot = median  # update global slot
            lineslots[k] = (widths[0], median, widths[-1])  # line slots
        return slot, lineslots

    # --------------------------------------------------------------------
    def joinligature(lig):
        """Return ligature character for a given pair / triple of characters.

        Args:
            lig: (str) 2/3 characters, e.g. "ff"
        Returns:
            Ligature, e.g. "ff" -> chr(0xFB00)
        """
        if lig == "ff":
            return chr(0xFB00)
        elif lig == "fi":
            return chr(0xFB01)
        elif lig == "fl":
            return chr(0xFB02)
        elif lig == "ft":
            return chr(0xFB05)
        elif lig == "st":
            return chr(0xFB06)
        elif lig == "ffi":
            return chr(0xFB03)
        elif lig == "ffl":
            return chr(0xFB04)
        return lig

    # --------------------------------------------------------------------
    def process_blocks(page, flags, blocks):
        left = page.rect.width  # left most used coordinate for TEXT
        right = 0  # rightmost coordinate for TEXT
        rowheight = page.rect.height  # smallest row height in use
        chars = []  # all chars and shapes here
        rows = set()  # bottom coordinates of lines

        # --- Now, process text blocks and ONLY update margins here ---
        if not blocks:
            blocks = page.get_text("rawdict", flags=flags)["blocks"]

        for block in blocks:
            for line in block["lines"]:
                if line["dir"] != (1, 0):
                    continue
                x0_line, y0_line, x1_line, y1_line = line["bbox"]
                if y1_line < 0 or y0_line > page.rect.height:
                    continue

                height = y1_line - y0_line
                if rowheight > height:
                    rowheight = height

                for span in line["spans"]:
                    if span["size"] <= fontsize:
                        continue
                    for c in span["chars"]:
                        x0_char, _, x1_char, _ = c["bbox"]
                        cwidth = x1_char - x0_char
                        ox, oy = c["origin"]
                        oy = int(round(oy))
                        rows.add(oy)
                        ch = c["c"]

                        # ### THIS IS THE CRITICAL CHANGE ###
                        # Only update left and right margins based on actual text characters.
                        if left > ox and ch.strip(
                        ):  # Use ch.strip() to ignore spaces
                            left = ox
                        if right < x1_char:
                            right = x1_char
                        # ### END CRITICAL CHANGE ###

                        # Handle ligatures (no changes here)
                        if cwidth == 0 and chars:
                            old_ch, old_ox, old_oy, old_cwidth = chars[-1]
                            if old_oy == oy:
                                if old_ch != chr(0xFB00):
                                    lig = joinligature(old_ch + ch)
                                elif ch == "i":
                                    lig = chr(0xFB03)
                                elif ch == "l":
                                    lig = chr(0xFB04)
                                else:
                                    lig = old_ch
                                chars[-1] = (lig, old_ox, old_oy, old_cwidth)
                                continue

                        chars.append((ch, ox, oy, cwidth))

        return rows, chars, rowheight, left, right

    # --------------------------------------------------------------------
    def make_textline(left, slot, lineslots, lchars):
        """Produce the text of one output line, correctly handling shapes."""
        minslot, median, maxslot = lineslots
        text = ""
        old_x1 = 0
        old_ox = 0

        for c in lchars:
            char, ox, _, cwidth = c

            ### MODIFICATION ###
            # Check if it's our special shape character
            is_shape = (char == "---")

            if is_shape:
                # For shapes, DO NOT subtract the left margin. Use absolute coordinates.
                absolute_ox = ox - left
                num_dashes = int(cwidth / (slot or 1))
                line_text = "-" * max(10, num_dashes)

                # Calculate spacing based on absolute position
                delta = int(absolute_ox / slot) - len(text)
                if delta > 0:
                    text += " " * delta
                text += line_text
                # After processing a shape, continue to the next item in the line
                continue
            ### END MODIFICATION ###

            # For regular text, apply normalization relative to the text margin
            ox_rel = ox - left
            x1_rel = ox_rel + cwidth

            if (old_ox <= ox_rel < old_x1 and char == text[-1]
                    and ox_rel - old_ox <= cwidth * 0.2):
                continue
            if char == " " and (old_x1 - ox_rel) / cwidth > 0.8:
                continue

            if ox_rel < old_x1 + minslot:
                text += char
                old_x1 = x1_rel
                old_ox = ox_rel
                continue

            delta = int(ox_rel / slot) - len(text)
            if delta > 1 and ox_rel <= old_x1 + slot * 2:
                delta = 1
            if ox_rel > old_x1 and delta >= 1:
                text += " " * delta

            text += char
            old_x1 = x1_rel
            old_ox = ox_rel
        return text.rstrip()

    # extract page text by single characters ("rawdict")
    rows, chars, rowheight, left, right = process_blocks(page, flags, blocks)
    if rows == set():
        if not skip_empty:
            textout.write(eop)  # write formfeed
        return
    # compute list of line coordinates - ignoring small (GRID) differences
    rows = curate_rows(rows, GRID)

    # sort all chars by x-coordinates, so every line will receive
    # them sorted.
    chars.sort(key=lambda c: c[1])

    # populate the lines with their char tuples
    lines, keys = make_lines(chars)

    slot, lineslots = compute_slots(keys, lines, right, left)

    # compute line advance in text output
    rowheight = rowheight * (rows[-1] - rows[0]) / \
        (rowheight * len(rows)) * 1.5
    rowpos = rows[0]  # first line positioned here
    textout.write(b"\n")
    for k in keys:  # walk through the lines
        while rowpos < k:  # honor distance between lines
            textout.write(b"\n")
            rowpos += rowheight
        text = make_textline(left, slot, lineslots[k], lines[k])
        textout.write((text + "\n").encode("utf8"))
        rowpos = k + rowheight

    textout.write(eop)


def gettext(page, blocks=None, extra_spaces=False):
    """
    Extracts text from a page object and returns it as a string, using a
    temporary file for intermediate storage.
    """
    flags = TEXT_PRESERVE_LIGATURES | TEXT_PRESERVE_WHITESPACE

    if not extra_spaces:  # In the original code, this logic was inverted.
        flags |= TEXT_INHIBIT_SPACES

    # Create a temporary file
    with tempfile.NamedTemporaryFile(mode="wb", delete=False) as temp_file:
        tmp_filename = temp_file.name
        # The original function passed several hardcoded variables.
        # These are now set here before calling page_layout.
        grid = 2
        fontsize = 3
        noformfeed = False
        skip_empty = False

        page_layout(page,
                    temp_file,
                    GRID=grid,
                    fontsize=fontsize,
                    noformfeed=noformfeed,
                    skip_empty=skip_empty,
                    flags=flags,
                    blocks=blocks)

    try:
        with open(tmp_filename, 'r', encoding='utf-8') as f:
            ret = f.read()
    finally:
        os.remove(tmp_filename)

    return ret
