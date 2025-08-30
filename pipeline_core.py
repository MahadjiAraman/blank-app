# -*- coding: utf-8 -*-
"""
Advanced PDF Processing Pipeline - Core Components
Extracted and adapted from user's pipeline for Streamlit integration
"""

from __future__ import annotations

import os
import json
import json5
import re
import time
import traceback
import random
import logging
import shutil
import io
from pathlib import Path
from datetime import datetime
from multiprocessing.dummy import Pool
from functools import partial
from typing import List, Dict, Any, Optional, Tuple, Type
from collections import defaultdict
from dataclasses import dataclass

import pymupdf as fitz  # PyMuPDF
import pandas as pd
from PIL import Image as PILImage

from google import genai
from google.genai import types

from proompts import (
    prompt_find_specid_and_page_type_v2,
    prompt_no_aggregation_with_schema,
    SpecSheetItem,
)
from get_text_layout import gettext
from add_json_to_pdf import add_json_to_pdf


# Configuration
class Config:
    """Centralized configuration class."""
    BASE_DIR: Path = Path(__file__).resolve().parent

    # Image filtering configuration (header/footer filtering)
    IMAGE_Y_MIN: float = 50.0  # Minimum Y coordinate to avoid headers
    IMAGE_Y_MAX: float = 750.0  # Maximum Y coordinate to avoid footers
    OUTPUT_DIR: Path = BASE_DIR / "llm_processed_results"
    OUTPUT_DIR.mkdir(exist_ok=True)

    DEFAULT_MODEL: str = 'gemini-2.5-flash'
    MAX_CHARS_PER_REQUEST: int = 60_000
    MAX_RETRIES: int = 3
    API_TIMEOUT_MS: int = 120_000
    BATCH_SIZE: int = 10
    FALLBACK_CHUNK_SIZE: int = 5
    PARALLEL_PROCESSES: int = 2
    IMAGE_Y_MIN: int = 100
    IMAGE_Y_MAX: int = 700


config = Config()


# LLM Processor with logging callback support
class LLMProcessor:
    """LLM service with logging callback support for UI integration."""

    def __init__(self, api_key: str, log_callback=None):
        self.api_key = api_key
        self.log_callback = log_callback or (lambda x: None)
        self.usage_stats = {'total_requests': 0, 'errors': 0}

    def log(self, message: str, level: str = "info"):
        """Log message with optional callback."""
        if level == "error":
            logging.error(message)
        elif level == "warning":
            logging.warning(message)
        else:
            logging.info(message)
        self.log_callback({
            "level": level,
            "message": message,
            "timestamp": datetime.now()
        })

    def _make_api_call(self, prompt: str, model: str,
                       schema: Optional[Type[Any]]) -> str:
        self.log(f"Making API call to model '{model}'")
        try:
            client = genai.Client(
                api_key=self.api_key,
                http_options=types.HttpOptions(timeout=config.API_TIMEOUT_MS))
            gen_config = types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=0))

            if schema:
                gen_config.response_mime_type = 'application/json'
                gen_config.response_schema = schema

            response = client.models.generate_content(model=model,
                                                      contents=prompt,
                                                      config=gen_config)
            self.usage_stats['total_requests'] += 1
            return response.text
        except Exception as e:
            self.usage_stats['errors'] += 1
            self.log(f"API call failed: {e}", "error")
            raise e

    def query(self,
              prompt_template: str,
              page_texts: List[str],
              model: str,
              schema: Optional[Type[Any]] = None,
              expect_json: bool = True,
              max_chars_override: Optional[int] = None) -> Optional[Any]:

        def _call_and_process(prompt_text: str) -> Optional[Any]:
            for attempt in range(config.MAX_RETRIES):
                try:
                    start_time = time.time()
                    response_str = self._make_api_call(prompt_text, model,
                                                       schema)
                    self.log(
                        f"API call successful. Time: {time.time() - start_time:.2f}s"
                    )

                    if not expect_json:
                        return response_str

                    json_str = response_str
                    if not schema:
                        match = re.search(r'```json\s*([\s\S]*?)\s*```',
                                          response_str, re.DOTALL)
                        if not match:
                            self.log(
                                f"Attempt {attempt + 1}: No JSON block found. Retrying...",
                                "warning")
                            time.sleep(2**attempt)
                            continue
                        json_str = match.group(1).strip()
                    return json5.loads(json_str)
                except Exception as e:
                    self.log(f"Attempt {attempt + 1} failed: {e}. Retrying...",
                             "warning")
                    time.sleep(2**attempt)
            self.log("All retry attempts failed", "error")
            return None

        char_limit = max_chars_override if max_chars_override is not None else config.MAX_CHARS_PER_REQUEST
        full_text = str(page_texts)

        if len(prompt_template % "") + len(full_text) <= char_limit:
            self.log(f"Request fits in single call ({len(full_text)} chars)")
            return _call_and_process(prompt_template % full_text)

        self.log(f"Request too large ({len(full_text)} chars). Splitting...",
                 "warning")
        combined_results: List[Any] = []
        for i in range(0, len(page_texts), config.FALLBACK_CHUNK_SIZE):
            chunk = page_texts[i:i + config.FALLBACK_CHUNK_SIZE]
            self.log(
                f"Processing sub-chunk {i // config.FALLBACK_CHUNK_SIZE + 1}..."
            )
            result = _call_and_process(prompt_template % str(chunk))
            if result:
                if isinstance(result, list):
                    combined_results.extend(result)
                else:
                    combined_results.append(result)
        return combined_results or None


# Data transformation functions
CANONICAL_KEY_RULES = [
    ('Approval Prior To Fabrication', 'Required Items'),
    ('Required Item', 'Required Items'),
]


def preprocess_keys(record: dict) -> dict:
    """Standardizes 'k' values using substring-based rules."""
    for key_list_name in ['attributes', 'metadata']:
        if key_list_name in record and record.get(key_list_name):
            for item in record[key_list_name]:
                if 'k' in item and item.get('k') is not None:
                    original_key = str(item['k'])
                    normalized_key = ' '.join(original_key.split()).title()
                    normalized_key = normalized_key.replace(
                        ' / ', '/').replace('/ ', '/').replace(' /', '/')

                    for substring, canonical_key in CANONICAL_KEY_RULES:
                        if substring.lower() in original_key.lower():
                            normalized_key = canonical_key
                            break

                    item['k'] = normalized_key
    return record


def _flatten_attributes_list(items_list: list) -> dict:
    """Flatten attribute list handling duplicate keys."""
    if not items_list:
        return {}

    flat_dict = {}
    key_counts = defaultdict(int)
    valid_items = [item for item in items_list if item.get('k')]

    for item in valid_items:
        key_counts[item['k']] += 1

    key_instance_counter = defaultdict(int)
    for item in valid_items:
        key, value = item['k'], item.get('v')
        if key_counts[key] > 1:
            key_instance_counter[key] += 1
            final_key = f"{key} {key_instance_counter[key]}"
        else:
            final_key = key
        flat_dict[final_key] = value

    return flat_dict


def merge_duplicate_pages(records_list: list) -> dict:
    """Merge duplicate records for the same page."""
    if not records_list:
        return {}
    if len(records_list) == 1:
        return records_list[0]

    primary_record = max(records_list,
                         key=lambda r: len(r.get('attributes', []))).copy()
    simple_text_keys = [
        'notes', 'instructions', 'product_type', 'product_category'
    ]

    for key in simple_text_keys:
        longest_val = None
        for rec in records_list:
            val = rec.get(key)
            if val is not None and (longest_val is None
                                    or len(str(val)) > len(str(longest_val))):
                longest_val = val
        primary_record[key] = longest_val

    all_attributes = defaultdict(list)
    for rec in records_list:
        for item in rec.get('attributes', []):
            if item.get('k'):
                all_attributes[item['k']].append(item.get('v'))

    final_attributes = []
    for k, v_list in all_attributes.items():
        valid_v_list = [v for v in v_list if v is not None]
        if valid_v_list:
            longest_v = max(valid_v_list, key=lambda v: len(str(v)))
            final_attributes.append({'k': k, 'v': longest_v})

    primary_record['attributes'] = final_attributes
    return primary_record


def transform_specs_to_dataframe(json_data: list) -> pd.DataFrame:
    """Transform spec JSON into DataFrame with preprocessing and merging."""
    if not json_data:
        return pd.DataFrame()

    # Preprocess keys
    preprocessed_data = [
        preprocess_keys(record.copy()) for record in json_data
    ]

    # Group and merge duplicate pages
    grouped_by_page = defaultdict(list)
    for record in preprocessed_data:
        if record.get('spec_id') is not None and record.get(
                'page_number') is not None:
            if record.get('page_type') in ('product_data',
                                           'continuation_page'):
                grouped_by_page[(record['spec_id'],
                                 record['page_number'])].append(record)

    merged_page_records = [
        merge_duplicate_pages(records) for records in grouped_by_page.values()
    ]

    # Group by spec_id for final aggregation
    grouped_specs = defaultdict(list)
    for record in merged_page_records:
        if record.get('spec_id'):
            grouped_specs[record['spec_id']].append(record)

    # Aggregate into final spec records
    aggregated_data = []
    for spec_id, records in grouped_specs.items():
        records.sort(key=lambda x: x.get('page_number', float('inf')))
        agg_record = {
            'spec_id': spec_id,
            'product_type': None,
            'product_category': None,
            'page_numbers': set(),
            'metadata': [],
            'attributes': [],
            'suppliers': [],
            'notes': [],
            'instructions': []
        }

        for record in records:
            if not agg_record['product_type'] and record.get('product_type'):
                agg_record['product_type'] = record.get('product_type')
            if not agg_record['product_category'] and record.get(
                    'product_category'):
                agg_record['product_category'] = record.get('product_category')
            if record.get('page_number') is not None:
                agg_record['page_numbers'].add(record.get('page_number'))

            agg_record['metadata'].extend(record.get('metadata', []))
            agg_record['attributes'].extend(record.get('attributes', []))
            agg_record['suppliers'].extend(record.get('suppliers', []))

            if record.get('notes'):
                agg_record['notes'].append(record['notes'])
            if record.get('instructions'):
                agg_record['instructions'].append(record['instructions'])

        agg_record['notes'] = "\n---\n".join(filter(None, agg_record['notes']))
        agg_record['instructions'] = "\n---\n".join(
            filter(None, agg_record['instructions']))
        agg_record['page_numbers'] = sorted(list(agg_record['page_numbers']))
        aggregated_data.append(agg_record)

    # Flatten data into DataFrame rows
    flat_rows = []
    for spec in aggregated_data:
        row = {
            'spec_id': spec['spec_id'],
            'product_type': spec['product_type'],
            'product_category': spec['product_category'],
            'page_numbers': ', '.join(map(str, spec['page_numbers']))
        }

        # Add metadata
        seen_meta_keys = set()
        for item in spec['metadata']:
            key = item.get('k')
            if key and key not in seen_meta_keys:
                row[key] = item.get('v')
                seen_meta_keys.add(key)

        # Deduplicate and flatten attributes
        unique_attributes_set = {
            tuple(sorted(d.items()))
            for d in spec.get('attributes', [])
            if 'k' in d and 'v' in d and d['v'] is not None
        }
        deduplicated_attributes = [dict(t) for t in unique_attributes_set]
        attr_flat = _flatten_attributes_list(deduplicated_attributes)
        row.update(attr_flat)

        # Add suppliers
        unique_suppliers = [
            dict(t) for t in
            {tuple(sorted(d.items()))
             for d in spec.get('suppliers', [])}
        ]
        for i, supplier in enumerate(unique_suppliers, 1):
            for key, value in supplier.items():
                if value:
                    row[f'Supplier {i} {key.replace("_", " ").title()}'] = value

        row['notes'] = spec['notes']
        row['instructions'] = spec['instructions']
        flat_rows.append(row)

    df = pd.DataFrame(flat_rows)
    if df.empty:
        return df

    # Final cleanup and column ordering
    core_cols = ['spec_id', 'product_type', 'product_category', 'page_numbers']
    meta_keys = {
        m['k']
        for spec in aggregated_data
        for m in spec.get('metadata', []) if m.get('k')
    }
    note_keys = {'notes', 'instructions'}
    meta_cols = sorted([c for c in df.columns if c in meta_keys])
    supplier_cols = sorted([c for c in df.columns if c.startswith('Supplier')])
    known_cols = set(core_cols + meta_cols + supplier_cols + list(note_keys))
    attr_cols = sorted([c for c in df.columns if c not in known_cols])
    final_ordered_cols = core_cols + meta_cols + attr_cols + supplier_cols + list(
        note_keys)
    final_ordered_cols = [
        col for col in final_ordered_cols if col in df.columns
    ]

    return df[final_ordered_cols]


# Pipeline classes
class PdfFile:
    """Initial PDF file state."""

    def __init__(self, path: str, log_callback=None):
        self.path = Path(path)
        self.log_callback = log_callback or (lambda x: None)
        if not self.path.exists():
            raise FileNotFoundError(f"PDF file not found at: {self.path}")
        self.log(f"Pipeline initialized for: {self.path.name}")

    def log(self, message: str, level: str = "info"):
        """Log with callback support."""
        if level == "error":
            logging.error(message)
        else:
            logging.info(message)
        self.log_callback({
            "level": level,
            "message": message,
            "timestamp": datetime.now()
        })

    def batch(self) -> 'BatchedPdf':
        self.log("Batching PDF into chunks...")
        doc = fitz.open(self.path)
        tasks: List[Dict[str, Any]] = []

        for i in range(0, len(doc), config.BATCH_SIZE):
            start_idx = (i - 1) if i > 0 else 0
            tasks.append({
                "start_page_idx": i,
                "pages_data": doc[start_idx:i + config.BATCH_SIZE]
            })

        self.log(f"Batched into {len(tasks)} tasks")
        return BatchedPdf(self.path, tasks, doc, self.log_callback)


class BatchedPdf:
    """PDF divided into batches."""

    def __init__(self,
                 path: Path,
                 tasks: List[Dict[str, Any]],
                 doc: fitz.Document,
                 log_callback=None):
        self.path = path
        self.tasks = tasks
        self.doc = doc
        self.log_callback = log_callback or (lambda x: None)

    def log(self, message: str, level: str = "info"):
        if level == "error":
            logging.error(message)
        else:
            logging.info(message)
        self.log_callback({
            "level": level,
            "message": message,
            "timestamp": datetime.now()
        })

    def _get_extraction_guide(self, llm_processor: LLMProcessor) -> str:
        self.log("Generating spec ID extraction guide...")
        doc_len = len(self.doc)

        if doc_len <= 10:
            sample_indices = list(range(doc_len))
        else:
            k_middle_sample = min(doc_len - 10, 40)
            sample_indices = (
                list(range(4)) +
                random.sample(range(4, doc_len - 5), k=k_middle_sample) +
                list(range(doc_len - 5, doc_len)))
            sample_indices = sorted(list(set(sample_indices)))

        page_texts = [
            f'\nPAGE {self.doc[i].number + 1}\n{gettext(self.doc[i])}'
            for i in sample_indices
        ]

        guide = llm_processor.query(prompt_find_specid_and_page_type_v2,
                                    page_texts,
                                    config.DEFAULT_MODEL,
                                    expect_json=False,
                                    max_chars_override=250_000)

        self.log("Extraction guide generated")
        return str(guide) if guide else "No specific guide generated."

    def _process_batch_worker(self, task: Dict[str, Any],
                              llm_processor: LLMProcessor,
                              extraction_guide: str,
                              pdf_stem: str) -> Dict[str, Any]:
        start_idx = task["start_page_idx"] + 1
        start_idx = start_idx - 1 if task["start_page_idx"] > 0 else start_idx
        output_path = config.OUTPUT_DIR / f"{pdf_stem}_chunk_{start_idx}.json"
        prompt_path = config.OUTPUT_DIR / "prompts" / f"{pdf_stem}_prompt_{start_idx}.txt"

        if output_path.exists():
            self.log(f"Chunk {start_idx} result already exists. Skipping.")
            return {
                "status": "skipped",
                "path": output_path,
                "prompt_path": prompt_path
            }

        self.log(f"Processing chunk starting at page {start_idx}...")

        page_texts = [
            f'\n*****PAGE {start_idx+n} STARTS HERE*****\n{gettext(p)} Page Number: {start_idx+n}\n*****PAGE {start_idx+n} ENDS HERE*****\n'
            for n, p in enumerate(task["pages_data"])
        ]

        prompt = prompt_no_aggregation_with_schema.format(
            specid_extraction_guide=extraction_guide, categories="")

        # Save prompt
        try:
            prompt_path.parent.mkdir(exist_ok=True, parents=True)
            with open(prompt_path, 'w', encoding='utf-8') as f:
                f.write(prompt % ''.join(page_texts))
        except Exception as e:
            self.log(f"Could not save prompt for chunk {start_idx}: {e}",
                     "warning")

        result_data = llm_processor.query(prompt,
                                          page_texts,
                                          config.DEFAULT_MODEL,
                                          schema=list[SpecSheetItem])

        if result_data:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, indent=2)
            self.log(f"Successfully processed chunk {start_idx}")
            return {
                "status": "success",
                "path": output_path,
                "prompt_path": prompt_path
            }

        self.log(f"Failed to process chunk {start_idx}", "error")
        return {"status": "error", "message": "LLM returned no valid data."}

    def process(self,
                llm_processor: LLMProcessor,
                progress_callback=None) -> 'ProcessedPdf':
        if not self.tasks:
            self.log("No batches to process", "warning")
            return ProcessedPdf(self.path, [], self.log_callback)

        extraction_guide = self._get_extraction_guide(llm_processor)
        results = []

        self.log(f"Starting processing of {len(self.tasks)} batches...")

        for i, task in enumerate(self.tasks):
            if progress_callback:
                progress_callback(i + 1, len(self.tasks))

            result = self._process_batch_worker(task, llm_processor,
                                                extraction_guide,
                                                self.path.stem)
            results.append(result)

        chunk_paths = [r['path'] for r in results if r.get('path')]
        self.log(
            f"Processing complete. Generated {len(chunk_paths)} result files")
        return ProcessedPdf(self.path, chunk_paths, self.log_callback)


class ProcessedPdf:
    """LLM results as raw JSON chunks."""

    def __init__(self, path: Path, chunk_paths: List[Path], log_callback=None):
        self.path = path
        self.chunk_file_paths = chunk_paths
        self.log_callback = log_callback or (lambda x: None)
        self.all_specs: List[Dict[str, Any]] = self._load_and_combine_specs()
        self.log(
            f"Combined {len(self.all_specs)} raw spec items from {len(chunk_paths)} files"
        )

    def log(self, message: str, level: str = "info"):
        if level == "error":
            logging.error(message)
        else:
            logging.info(message)
        self.log_callback({
            "level": level,
            "message": message,
            "timestamp": datetime.now()
        })

    def _load_and_combine_specs(self) -> List[Dict[str, Any]]:
        all_specs: List[Dict[str, Any]] = []
        for file_path in self.chunk_file_paths:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json5.load(f)
                    if isinstance(data, list):
                        all_specs.extend(data)
            except Exception as e:
                self.log(f"Could not read chunk {file_path.name}: {e}",
                         "warning")
        return all_specs

    def aggregate_and_transform(self) -> 'AggregatedPdf':
        self.log("Transforming raw data into structured DataFrame...")
        df = transform_specs_to_dataframe(self.all_specs)
        self.log(f"Transformation complete. DataFrame: {len(df)} rows")
        return AggregatedPdf(self.path, df, self.all_specs)


class AggregatedPdf:
    """State 4: Data aggregated into a structured DataFrame."""

    def __init__(self, path: Path, dataframe: pd.DataFrame,
                 all_specs: List[Dict[str, Any]]):
        self.path = path
        self.dataframe = dataframe
        self.all_specs = all_specs

    def enrich_with_images(self, one_based_idx=True) -> 'EnrichedPdf':
        """Add image enrichment step to the pipeline."""
        logging.info("Stage: Enriching DataFrame with extracted images...")
        if self.dataframe.empty:
            logging.warning("DataFrame is empty, skipping image enrichment.")
            return EnrichedPdf(self.path, self.dataframe, self.all_specs,
                               Path("empty_images"))

        images_dir = Path(
            f"llm_processed_results/{self.path.stem}_images_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        )
        images_dir.mkdir(parents=True, exist_ok=True)

        try:
            doc = fitz.open(self.path)

            def _concat_images_vertically(
                    images: List[PILImage.Image]) -> PILImage.Image:
                """Concatenate multiple PIL images vertically."""
                if not images:
                    raise ValueError("Cannot concatenate empty list of images")
                if len(images) == 1:
                    return images[0]

                # Calculate total width and height
                total_width = max(img.width for img in images)
                total_height = sum(img.height for img in images)

                # Create new image
                concatenated = PILImage.new('RGB', (total_width, total_height),
                                            color='white')

                # Paste images vertically
                y_offset = 0
                for img in images:
                    # Center smaller images horizontally
                    x_offset = (total_width - img.width) // 2
                    concatenated.paste(img, (x_offset, y_offset))
                    y_offset += img.height

                return concatenated

            def _extract_images_for_row(row: pd.Series,
                                        one_based_idx=True) -> str:
                spec_id = row.get('spec_id', 'unknown_spec')
                page_numbers_str = str(row.get('page_numbers', ''))
                if not page_numbers_str:
                    return ""

                try:
                    page_numbers = [
                        int(p.strip()) for p in page_numbers_str.split(',')
                    ]
                except ValueError:
                    return ""

                final_image_filenames = []

                for page_num in page_numbers:
                    if page_num <= len(doc):
                        try:
                            if one_based_idx:
                                page = doc[page_num - 1]  # convert to 0-based
                            else:
                                page = doc[page_num]

                            if page is None:
                                continue

                            # Collect all valid PIL images on the current page with filtering
                            images_on_this_page: List[PILImage.Image] = []
                            for img_info in page.get_image_info(xrefs=True):
                                img_bbox = img_info['bbox']

                                # Apply vertical region filter (header/footer filtering)
                                if Config.IMAGE_Y_MIN <= img_bbox[
                                        1] and img_bbox[
                                            3] <= Config.IMAGE_Y_MAX:
                                    try:
                                        xref = img_info['xref']
                                        img_bytes = doc.extract_image(
                                            xref)["image"]
                                        pil_image = PILImage.open(
                                            io.BytesIO(img_bytes))
                                        images_on_this_page.append(pil_image)
                                    except Exception as e:
                                        logging.warning(
                                            f"Could not extract image xref {xref} for spec '{spec_id}' on page {page_num}: {e}"
                                        )

                            if not images_on_this_page:
                                continue

                            # Determine final image for the page (single or concatenated)
                            final_image_for_page: PILImage.Image
                            if len(images_on_this_page) > 1:
                                logging.info(
                                    f"Concatenating {len(images_on_this_page)} images for spec '{spec_id}' on page {page_num}."
                                )
                                final_image_for_page = _concat_images_vertically(
                                    images_on_this_page)
                            else:
                                final_image_for_page = images_on_this_page[0]

                            # Save the single (potentially composite) image for the page
                            img_filename = f"p{page_num}_{spec_id}_combined.png"
                            img_path = images_dir / img_filename
                            final_image_for_page.save(str(img_path), "PNG")
                            final_image_filenames.append(img_filename)

                        except Exception as e:
                            logging.warning(
                                f"Error processing page {page_num} for spec '{spec_id}': {e}"
                            )

                return ','.join(final_image_filenames)

            # Add images column to dataframe
            if 'images' not in self.dataframe.columns:
                self.dataframe['images'] = self.dataframe.apply(
                    _extract_images_for_row,
                    axis=1,
                    one_based_idx=one_based_idx)

            doc.close()
            logging.info(
                f"Image enrichment complete. Images saved to '{images_dir.name}'."
            )

        except Exception as e:
            logging.error(f"Error during image enrichment: {e}")
            # Continue without images
            if 'images' not in self.dataframe.columns:
                self.dataframe['images'] = ""

        return EnrichedPdf(self.path, self.dataframe, self.all_specs,
                           images_dir)


class EnrichedPdf:
    """State 5: DataFrame is enriched with extracted image paths."""

    def __init__(self, path: Path, dataframe: pd.DataFrame,
                 all_specs: List[Dict[str, Any]], images_dir: Path):
        self.path = path
        self.dataframe = dataframe
        self.all_specs = all_specs
        self.images_dir = images_dir

    def archive_results(self, one_based_idx=True) -> 'ArchivedPackage':
        """Create final archive with all results."""
        logging.info("Stage: Archiving all generated artifacts...")
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        archive_work_dir = Path(
            f"llm_processed_results/{self.path.stem}_temp_archive_{timestamp}")
        archive_work_dir.mkdir(parents=True, exist_ok=True)

        try:
            # 1. Save final enriched DataFrame
            final_csv_path = archive_work_dir / f"{self.path.stem}_FINAL.csv"
            self.dataframe.sort_values(
                'page_numbers',
                key=lambda x:
                (x.astype(str).str.split(',').str[0].astype(int))).to_csv(
                    final_csv_path, index=False)
            logging.info(f"Saved final CSV to archive directory.")

            if self.all_specs:
                # 2. Save page-level test DataFrame
                df_test = pd.DataFrame(
                    [(i.get('spec_id'), i.get('page_number'),
                      i.get('page_type'))
                     for i in self.all_specs if isinstance(i, dict)],
                    columns=['spec_id', 'page_number', 'page_type'])
                df_test.to_csv(archive_work_dir /
                               f"{self.path.stem}_page_level_debug.csv",
                               index=False)
                logging.info("Saved page-level debug CSV.")

                # 3. Save raw JSON data
                with open(archive_work_dir / f"{self.path.stem}_raw_data.json",
                          'w',
                          encoding='utf-8') as f:
                    json.dump(self.all_specs, f, indent=2, default=str)
                logging.info("Saved raw JSON data.")

            # 4. Create annotated PDF
            annotated_pdf_path = archive_work_dir / f"{self.path.stem}_annotated.pdf"
            try:
                # Try to import and use the add_json_to_pdf function
                from add_json_to_pdf import add_json_to_pdf
                add_json_to_pdf(input_pdf=str(self.path),
                                json_results={'v1': self.all_specs},
                                output_pdf=str(annotated_pdf_path),
                                one_based_idx=one_based_idx)
                logging.info("Created annotated PDF.")
            except Exception as e:
                logging.warning(f"Could not create annotated PDF: {e}")
                # Create a simple copy instead
                shutil.copy2(self.path, annotated_pdf_path)

            # 5. Move images folder into archive directory if it exists
            if self.images_dir.exists() and any(self.images_dir.iterdir()):
                target_images_dir = archive_work_dir / self.images_dir.name
                shutil.move(str(self.images_dir), str(target_images_dir))
                logging.info("Moved images folder to archive directory.")

            # 6. Create the zip archive
            archive_base_path = Path(
                f"llm_processed_results/{self.path.stem}_output_{timestamp}")
            archive_path = shutil.make_archive(str(archive_base_path), 'zip',
                                               str(archive_work_dir))
            logging.info(
                f"Successfully created archive: {Path(archive_path).name}")

        finally:
            # 7. Clean up the temporary working directory
            shutil.rmtree(archive_work_dir)
            logging.info(f"Cleaned up temporary archive directory.")

        return ArchivedPackage(self.path, Path(archive_path))


class ArchivedPackage:
    """Final State: All results have been packaged into a zip archive."""

    def __init__(self,
                 original_path: Path,
                 archive_path: Path,
                 log_callback=None):
        self.original_path = original_path
        self.archive_path = archive_path
        logging.info(
            f"PIPELINE COMPLETE for {original_path.name}. Final output: {archive_path.name}"
        )
        self.log_callback = log_callback or (lambda x: None)

    def log(self, message: str, level: str = "info"):
        if level == "error":
            logging.error(message)
        else:
            logging.info(message)
        self.log_callback({
            "level": level,
            "message": message,
            "timestamp": datetime.now()
        })

    def save_results(self) -> Path:
        """Save all results to an archive."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_dir = config.OUTPUT_DIR / f"{self.path.stem}_results_{timestamp}"
        results_dir.mkdir(exist_ok=True)

        # Save DataFrame as CSV
        csv_path = results_dir / f"{self.path.stem}_final.csv"
        self.dataframe.to_csv(csv_path, index=False)

        # Save raw JSON
        json_path = results_dir / f"{self.path.stem}_raw_specs.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.all_specs, f, indent=2)

        # Copy chunk files
        chunks_dir = results_dir / "chunks"
        chunks_dir.mkdir(exist_ok=True)

        for chunk_file in config.OUTPUT_DIR.glob(
                f"{self.path.stem}_chunk_*.json"):
            shutil.copy2(chunk_file, chunks_dir)

        # Copy prompts
        prompts_dir = results_dir / "prompts"
        prompts_source = config.OUTPUT_DIR / "prompts"
        if prompts_source.exists():
            shutil.copytree(prompts_source, prompts_dir, dirs_exist_ok=True)

        self.log(f"Results saved to: {results_dir}")
        return results_dir
