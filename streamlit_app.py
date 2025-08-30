import streamlit as st
import pymupdf as fitz  # PyMuPDF
import json
import pandas as pd
import io
import os
import zipfile
import shutil
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import time
import threading
from queue import Queue

from pipeline_core import (PdfFile, LLMProcessor, Config, config,
                          AggregatedPdf, ProcessedPdf, BatchedPdf)



# Custom logging handler to capture all logs
class StreamlitLogHandler(logging.Handler):
    """Custom logging handler that stores logs in session state for download."""

    def __init__(self):
        super().__init__()
        self.setLevel(logging.DEBUG)

    def emit(self, record):
        try:
            if 'all_logging_messages' not in st.session_state:
                st.session_state.all_logging_messages = []

            log_entry = {
                'timestamp':
                datetime.fromtimestamp(record.created).isoformat(),
                'level': record.levelname.lower(),
                'module': record.module,
                'function': record.funcName,
                'line': record.lineno,
                'message': self.format(record)
            }
            st.session_state.all_logging_messages.append(log_entry)

            # Keep only last 1000 log entries to prevent memory issues
            if len(st.session_state.all_logging_messages) > 1000:
                st.session_state.all_logging_messages = st.session_state.all_logging_messages[
                    -1000:]
        except Exception:
            pass  # Ignore errors in logging handler


# Available models and API key management
AVAILABLE_MODELS = [
    'gemini-2.5-flash', 'gemini-2.5-flash-lite-preview-06-17',
    'gemini-1.5-pro', 'gemini-1.5-flash'
]


class APIKeyManager:
    """Manages multiple API keys with usage tracking and persistence."""

    def __init__(self):
        self.usage_file = Path("api_usage.json")
        self.load_usage_stats()

    def load_usage_stats(self):
        """Load usage statistics from file."""
        if self.usage_file.exists():
            try:
                with open(self.usage_file, 'r') as f:
                    self.usage_stats = json.load(f)
            except:
                self.usage_stats = {}
        else:
            self.usage_stats = {}

    def save_usage_stats(self):
        """Save usage statistics to file."""
        try:
            with open(self.usage_file, 'w') as f:
                json.dump(self.usage_stats, f, indent=2, default=str)
        except Exception as e:
            st.warning(f"Could not save usage stats: {e}")

    def get_key_stats(self, api_key: str) -> Dict:
        """Get usage statistics for a specific API key."""
        key_id = api_key[-8:] if len(api_key) > 8 else api_key
        if key_id not in self.usage_stats:
            self.usage_stats[key_id] = {
                'total_requests': 0,
                'successful_requests': 0,
                'failed_requests': 0,
                'models': {},
                'last_used': None,
                'created': datetime.now().isoformat()
            }
        return self.usage_stats[key_id]

    def record_usage(self,
                     api_key: str,
                     model: str,
                     success: bool = True,
                     response_time: float = 0):
        """Record API usage for tracking."""
        stats = self.get_key_stats(api_key)
        stats['total_requests'] += 1
        stats['last_used'] = datetime.now().isoformat()

        if success:
            stats['successful_requests'] += 1
        else:
            stats['failed_requests'] += 1

        # Track per-model usage
        if model not in stats['models']:
            stats['models'][model] = {
                'requests': 0,
                'successes': 0,
                'failures': 0,
                'avg_response_time': 0
            }

        model_stats = stats['models'][model]
        model_stats['requests'] += 1
        if success:
            model_stats['successes'] += 1
        else:
            model_stats['failures'] += 1

        # Update average response time
        if response_time > 0:
            current_avg = model_stats['avg_response_time']
            model_stats['avg_response_time'] = (
                current_avg * (model_stats['requests'] - 1) +
                response_time) / model_stats['requests']

        self.save_usage_stats()

    def get_best_key(self, available_keys: List[str]) -> str:
        """Get the best available key based on usage statistics."""
        if not available_keys:
            raise ValueError("No API keys available")

        if len(available_keys) == 1:
            return available_keys[0]

        # Calculate scores for each key (lower is better)
        key_scores = []
        for key in available_keys:
            stats = self.get_key_stats(key)

            # Calculate failure rate
            total_requests = stats['total_requests']
            failure_rate = stats['failed_requests'] / max(total_requests, 1)

            # Calculate recent usage (prefer less recently used keys)
            if stats['last_used']:
                last_used = datetime.fromisoformat(stats['last_used'])
                hours_since_use = (datetime.now() -
                                   last_used).total_seconds() / 3600
                recency_score = 1 / max(hours_since_use,
                                        0.1)  # Lower is better
            else:
                recency_score = 0  # Never used = best

            # Combined score (lower is better)
            score = failure_rate * 10 + recency_score + total_requests * 0.01
            key_scores.append((score, key))

        # Return key with lowest score
        return min(key_scores)[1]


# Initialize API key manager
@st.cache_resource
def get_api_key_manager():
    return APIKeyManager()


class EnhancedLLMProcessor(LLMProcessor):
    """Enhanced LLM Processor with usage tracking integration."""

    def __init__(self,
                 api_key: str,
                 model: str,
                 api_key_manager: APIKeyManager,
                 log_callback=None):
        self.api_key = api_key
        self.model = model
        self.api_key_manager = api_key_manager
        self.log_callback = log_callback or (lambda x: None)
        super().__init__(api_key, log_callback)

    def _make_api_call(self,
                       prompt: str,
                       model: str,
                       schema: Optional[type] = None) -> str:
        start_time = time.time()
        try:
            response = super()._make_api_call(prompt, model, schema)
            response_time = time.time() - start_time
            self.api_key_manager.record_usage(self.api_key, model, True,
                                              response_time)
            return response
        except Exception as e:
            response_time = time.time() - start_time
            self.api_key_manager.record_usage(self.api_key, model, False,
                                              response_time)
            raise e


def create_zip_archive(results_dir: Path) -> Path:
    """Create a downloadable zip archive of all results."""
    zip_path = results_dir.parent / f"{results_dir.name}.zip"

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path in results_dir.rglob('*'):
            if file_path.is_file():
                arcname = file_path.relative_to(results_dir)
                zipf.write(file_path, arcname)

    return zip_path


def display_api_usage_stats(api_key_manager: APIKeyManager,
                            available_keys: List[str]):
    """Display comprehensive API usage statistics."""
    if not available_keys:
        st.warning("No API keys configured")
        return

    st.subheader("📊 API Usage Statistics")

    # Summary stats
    total_requests = 0
    total_successes = 0
    total_failures = 0

    for key in available_keys:
        stats = api_key_manager.get_key_stats(key)
        total_requests += stats['total_requests']
        total_successes += stats['successful_requests']
        total_failures += stats['failed_requests']

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Requests", total_requests)
    with col2:
        st.metric("Successes", total_successes)
    with col3:
        st.metric("Failures", total_failures)
    with col4:
        success_rate = (total_successes / max(total_requests, 1)) * 100
        st.metric("Success Rate", f"{success_rate:.1f}%")

    # Per-key details
    if st.checkbox("Show detailed key statistics"):
        for key in available_keys:
            key_display = f"Key ending in ...{key[-8:]}"
            stats = api_key_manager.get_key_stats(key)

            with st.expander(f"🔑 {key_display}"):
                col1, col2 = st.columns(2)

                with col1:
                    st.write("**General Stats:**")
                    st.write(f"Total Requests: {stats['total_requests']}")
                    st.write(
                        f"Success Rate: {(stats['successful_requests']/max(stats['total_requests'],1)*100):.1f}%"
                    )
                    if stats['last_used']:
                        last_used = datetime.fromisoformat(stats['last_used'])
                        st.write(
                            f"Last Used: {last_used.strftime('%Y-%m-%d %H:%M:%S')}"
                        )

                with col2:
                    st.write("**Model Usage:**")
                    for model, model_stats in stats['models'].items():
                        success_rate = (model_stats['successes'] /
                                        max(model_stats['requests'], 1)) * 100
                        avg_time = model_stats['avg_response_time']
                        st.write(
                            f"{model}: {model_stats['requests']} req ({success_rate:.1f}% success, {avg_time:.2f}s avg)"
                        )


def display_processing_logs(logs: List[Dict]):
    """Display processing logs in a structured format."""
    if not logs:
        return

    st.subheader("📋 Processing Logs")

    # Filter controls and download button
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        log_level_filter = st.selectbox("Filter by level:",
                                        ["All", "info", "warning", "error"])
    with col2:
        show_recent_only = st.checkbox("Show only recent (last 50)",
                                       value=True)
    with col3:
        # Download all logs as text file
        if logs:
            logs_text = ""
            for log in logs:
                timestamp = log.get("timestamp", datetime.now())
                if isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(
                        timestamp.replace('Z', '+00:00'))
                level = log.get("level", "info").upper()
                message = log.get("message", "")
                logs_text += f"[{timestamp.strftime('%Y-%m-%d %H:%M:%S')}] {level}: {message}\n"

            st.download_button(
                label="📄 Download Logs",
                data=logs_text,
                file_name=
                f"processing_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain")

    # Filter logs
    filtered_logs = logs
    if log_level_filter != "All":
        filtered_logs = [
            log for log in logs if log.get("level") == log_level_filter
        ]

    if show_recent_only:
        filtered_logs = filtered_logs[-50:]

    # Display logs
    log_container = st.container()
    with log_container:
        for log in filtered_logs:
            timestamp = log.get("timestamp", datetime.now())
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(
                    timestamp.replace('Z', '+00:00'))

            level = log.get("level", "info")
            message = log.get("message", "")

            # Color code by level
            if level == "error":
                st.error(f"🔴 {timestamp.strftime('%H:%M:%S')} - {message}")
            elif level == "warning":
                st.warning(f"🟡 {timestamp.strftime('%H:%M:%S')} - {message}")
            else:
                st.info(f"🔵 {timestamp.strftime('%H:%M:%S')} - {message}")


def display_batch_progress(current_batch: int, total_batches: int,
                           batch_logs: Dict):
    """Display real-time batch processing progress."""
    if total_batches == 0:
        return

    st.subheader(
        f"⚙️ Batch Processing Progress ({current_batch}/{total_batches})")

    # Overall progress
    progress = current_batch / total_batches
    st.progress(progress)

    # Batch status grid
    cols_per_row = 10
    batch_rows = (total_batches + cols_per_row - 1) // cols_per_row

    for row in range(batch_rows):
        cols = st.columns(cols_per_row)
        for col_idx in range(cols_per_row):
            batch_idx = row * cols_per_row + col_idx
            if batch_idx < total_batches:
                with cols[col_idx]:
                    batch_num = batch_idx + 1
                    if batch_num <= current_batch:
                        if batch_num in batch_logs and batch_logs[
                                batch_num].get('error'):
                            st.error(f"❌ {batch_num}")
                        else:
                            st.success(f"✅ {batch_num}")
                    else:
                        st.info(f"⏳ {batch_num}")


def display_json_viewer(json_data: Any, title: str = "JSON Data"):
    """Display JSON data with syntax highlighting and search."""
    st.subheader(f"🔍 {title}")

    if not json_data:
        st.info("No JSON data available")
        return

    # Search functionality
    search_term = st.text_input("Search in JSON:",
                                placeholder="Enter search term...")

    # Convert to string for display
    json_str = json.dumps(json_data, indent=2, default=str)

    # Highlight search terms
    if search_term and search_term.strip():
        lines = json_str.split('\n')
        highlighted_lines = []
        for line in lines:
            if search_term.lower() in line.lower():
                highlighted_lines.append(f"**{line}**")
            else:
                highlighted_lines.append(line)
        json_str = '\n'.join(highlighted_lines)

    # Display with scroll
    st.code(json_str, language='json')

    # Download option
    st.download_button(label="📥 Download JSON",
                       data=json.dumps(json_data, indent=2, default=str),
                       file_name=f"{title.lower().replace(' ', '_')}.json",
                       mime="application/json")


def display_prompts_viewer(results_dir: Path):
    """Display generated prompts for inspection."""
    prompts_dir = results_dir / "prompts"
    if not prompts_dir.exists():
        st.info("No prompts available")
        return

    st.subheader("📝 Generated Prompts")

    prompt_files = list(prompts_dir.glob("*.txt"))
    if not prompt_files:
        st.info("No prompt files found")
        return

    # Select prompt file
    selected_file = st.selectbox("Select prompt file:",
                                 prompt_files,
                                 format_func=lambda x: x.name)

    if selected_file:
        try:
            with open(selected_file, 'r', encoding='utf-8') as f:
                prompt_content = f.read()

            st.text_area("Prompt Content:", prompt_content, height=400)

            # Download option
            st.download_button(label="📥 Download Prompt",
                               data=prompt_content,
                               file_name=selected_file.name,
                               mime="text/plain")
        except Exception as e:
            st.error(f"Could not read prompt file: {e}")


def main():
    st.set_page_config(page_title="Advanced PDF Processing Pipeline",
                       page_icon="📄",
                       layout="wide",
                       initial_sidebar_state="expanded")

    st.title("🚀 Advanced PDF Processing Pipeline")
    st.markdown(
        "Upload PDF documents for AI-powered extraction and conversion to structured data"
    )

    # Initialize components
    api_key_manager = get_api_key_manager()

    # Setup comprehensive logging capture
    if 'logging_setup' not in st.session_state:
        # Create and configure the logging handler
        log_handler = StreamlitLogHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        log_handler.setFormatter(formatter)

        # Add handler to root logger to capture all logs
        root_logger = logging.getLogger()
        root_logger.addHandler(log_handler)
        root_logger.setLevel(logging.DEBUG)

        st.session_state.logging_setup = True
        st.session_state.all_logging_messages = []

    # Initialize session state
    session_keys = [
        'processing_logs', 'current_batch', 'total_batches', 'batch_logs',
        'processing_complete', 'results_data', 'raw_json_data', 'results_dir',
        'archive_path', 'all_logging_messages'
    ]

    for key in session_keys:
        if key not in st.session_state:
            if key == 'processing_logs':
                st.session_state[key] = []
            elif key in ['current_batch', 'total_batches']:
                st.session_state[key] = 0
            elif key == 'batch_logs':
                st.session_state[key] = {}
            elif key == 'processing_complete':
                st.session_state[key] = False
            else:
                st.session_state[key] = None

    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        #os.environ['GEMINI_API_KEY'] = 'test'
        # API Key management
        st.subheader("🔑 API Keys")

        # Get available keys from environment
        available_keys = []
        for i in range(1, 6):  # Support up to 5 keys
            key_env_name = f"GEMINI_API_KEY_{i}" if i > 1 else "GEMINI_API_KEY"
            key_value = os.environ.get(key_env_name)
            if key_value:
                available_keys.append(key_value)

        if not available_keys:
            st.error(
                "No API keys found! Please add GEMINI_API_KEY to your environment."
            )
            st.stop()

        # Key selection
        if len(available_keys) > 1:
            key_options = [f"Auto-select best key"] + [
                f"Key ...{key[-8:]}" for key in available_keys
            ]
            selected_key_index = st.selectbox(
                "Select API Key:",
                range(len(key_options)),
                format_func=lambda x: key_options[x])

            if selected_key_index == 0:
                selected_key = api_key_manager.get_best_key(available_keys)
                st.info(f"Using key: ...{selected_key[-8:]}")
            else:
                selected_key = available_keys[selected_key_index - 1]
        else:
            selected_key = available_keys[0]
            st.info(f"Using key: ...{selected_key[-8:]}")

        # Model selection
        st.subheader("🤖 Model Configuration")
        selected_model = st.selectbox("Select Model:",
                                      AVAILABLE_MODELS,
                                      index=0)

        # Processing configuration
        st.subheader("⚙️ Processing Settings")
        batch_size = st.slider("Batch Size (pages):", 1, 20, config.BATCH_SIZE)
        parallel_processes = st.slider("Parallel Processes:", 1, 4,
                                       config.PARALLEL_PROCESSES)
        max_retries = st.slider("Max Retries:", 1, 5, config.MAX_RETRIES)

        # Update config
        config.BATCH_SIZE = batch_size
        config.PARALLEL_PROCESSES = parallel_processes
        config.MAX_RETRIES = max_retries

        st.divider()

        # Display API usage stats
        display_api_usage_stats(api_key_manager, available_keys)

    # Main interface tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📤 Upload & Process", "📊 Results & Data", "📋 Logs & Progress",
        "🔍 JSON Viewer", "📝 Prompts & Debug"
    ])

    with tab1:
        st.header("📤 PDF Upload and Processing")

        # File upload
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=['pdf'],
            help="Upload a PDF document for advanced AI processing")

        if uploaded_file is not None:
            st.success(f"📁 Uploaded: {uploaded_file.name}")

            # Processing controls
            col1, col2 = st.columns(2)
            with col1:
                process_button = st.button("🚀 Start Processing",
                                           type="primary",
                                           use_container_width=True)
            with col2:
                if st.session_state.processing_complete:
                    if st.button("🔄 Process Again", use_container_width=True):
                        # Reset state for reprocessing
                        for key in session_keys:
                            if key == 'processing_logs':
                                st.session_state[key] = []
                            elif key in ['current_batch', 'total_batches']:
                                st.session_state[key] = 0
                            elif key == 'batch_logs':
                                st.session_state[key] = {}
                            else:
                                st.session_state[
                                    key] = False if key == 'processing_complete' else None
                        st.rerun()

            if process_button:
                try:
                    # Clear previous session data to avoid mixing results
                    keys_to_clear = [
                        'results_data', 'raw_json_data', 'results_dir',
                        'archive_path', 'processing_logs', 'batch_logs',
                        'processing_complete'
                    ]
                    for key in keys_to_clear:
                        if key in st.session_state:
                            if key == 'processing_logs':
                                st.session_state[key] = []
                            elif key == 'batch_logs':
                                st.session_state[key] = {}
                            elif key == 'processing_complete':
                                st.session_state[key] = False
                            else:
                                del st.session_state[key]

                    # Create unique working directory using source PDF name
                    pdf_stem = Path(uploaded_file.name).stem
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    work_dir = Path(
                        f"llm_processed_results/{pdf_stem}_{timestamp}")
                    work_dir.mkdir(parents=True, exist_ok=True)

                    # Save uploaded file with original name
                    pdf_path = work_dir / uploaded_file.name
                    with open(pdf_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    # Initialize processing pipeline
                    def log_callback(log_entry):
                        st.session_state.processing_logs.append(log_entry)

                    def progress_callback(current, total):
                        st.session_state.current_batch = current
                        st.session_state.total_batches = total

                    # Create enhanced LLM processor
                    llm_processor = EnhancedLLMProcessor(
                        selected_key, selected_model, api_key_manager,
                        log_callback)

                    # Process with progress tracking
                    progress_container = st.container()

                    with progress_container:
                        st.info("🔄 Starting PDF processing pipeline...")

                        # Stage 1: Initialize PDF
                        pdf_file = PdfFile(str(pdf_path), log_callback)
                        st.success("✅ PDF loaded successfully")

                        # Stage 2: Batch PDF
                        batched_pdf = pdf_file.batch()
                        st.success(
                            f"✅ PDF batched into {len(batched_pdf.tasks)} chunks"
                        )
                        st.session_state.total_batches = len(batched_pdf.tasks)

                        # Stage 3: Process batches
                        st.info("🤖 Processing with AI...")
                        processed_pdf = batched_pdf.process(
                            llm_processor, progress_callback)
                        st.success("✅ AI processing completed")

                        # Stage 4: Aggregate and transform
                        st.info("📊 Transforming data...")
                        aggregated_pdf = processed_pdf.aggregate_and_transform(
                        )
                        st.success("✅ Data transformation completed")

                        # Stage 5: Image enrichment
                        st.info("🖼️ Enriching with images...")
                        enriched_pdf = aggregated_pdf.enrich_with_images()
                        st.success("✅ Image enrichment completed")

                        # Stage 6: Create final archive
                        st.info("📦 Creating final archive...")
                        archived_package = enriched_pdf.archive_results()
                        st.success("✅ Archive created")

                        # Update session state
                        st.session_state.processing_complete = True
                        st.session_state.results_data = enriched_pdf.dataframe
                        st.session_state.raw_json_data = enriched_pdf.all_specs
                        st.session_state.results_dir = work_dir
                        st.session_state.archive_path = archived_package.archive_path

                        # Clean up temp file
                        #temp_pdf_path.unlink()

                        st.success(
                            "🎉 Processing pipeline completed successfully!")
                        st.balloons()

                except Exception as e:
                    st.error(f"❌ Processing failed: {str(e)}")
                    st.exception(e)

    with tab2:
        st.header("📊 Results and Data")

        if st.session_state.processing_complete and st.session_state.results_data is not None:
            df = st.session_state.results_data

            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Records", len(df))
            with col2:
                st.metric("Columns", len(df.columns))
            with col3:
                unique_specs = df['spec_id'].nunique(
                ) if 'spec_id' in df.columns else 0
                st.metric("Unique Specs", unique_specs)
            with col4:
                total_pages = df['page_numbers'].str.split(',').explode(
                ).nunique() if 'page_numbers' in df.columns else 0
                st.metric("Total Pages", total_pages)

            # Data preview
            st.subheader("📋 Data Preview")

            # Search and filter
            col1, col2 = st.columns(2)
            with col1:
                search_term = st.text_input("🔍 Search in data:",
                                            placeholder="Enter search term...")
            with col2:
                show_all_columns = st.checkbox("Show all columns", value=False)

            # Apply search filter
            display_df = df
            if search_term:
                mask = df.astype(str).apply(lambda x: x.str.contains(
                    search_term, case=False, na=False)).any(axis=1)
                display_df = df[mask]

            # Column selection
            if not show_all_columns:
                important_cols = [
                    'spec_id', 'product_type', 'product_category',
                    'page_numbers'
                ]
                available_cols = [
                    col for col in important_cols if col in df.columns
                ]
                if len(df.columns) > len(available_cols):
                    remaining_cols = [
                        col for col in df.columns if col not in available_cols
                    ]
                    available_cols.extend(remaining_cols[:8 -
                                                         len(available_cols)])
                # Remove duplicates while preserving order
                available_cols = list(dict.fromkeys(available_cols))
                display_df = display_df[available_cols]

            # Display data
            st.dataframe(display_df, use_container_width=True, height=400)

            # Download options
            st.subheader("📥 Download Options")
            col1, col2, col3 = st.columns(3)

            with col1:
                csv_data = df.to_csv(index=False)
                st.download_button(
                    label="📊 Download CSV",
                    data=csv_data,
                    file_name=
                    f"processed_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv")

            with col2:
                excel_buffer = io.BytesIO()
                df.to_excel(excel_buffer, index=False, engine='openpyxl')
                excel_data = excel_buffer.getvalue()
                st.download_button(
                    label="📈 Download Excel",
                    data=excel_data,
                    file_name=
                    f"processed_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime=
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

            with col3:
                if st.session_state.get('archive_path') and Path(
                        st.session_state.archive_path).exists():
                    with open(st.session_state.archive_path, 'rb') as f:
                        zip_data = f.read()
                    st.download_button(label="📦 Download Complete Archive",
                                       data=zip_data,
                                       file_name=Path(
                                           st.session_state.archive_path).name,
                                       mime="application/zip")
        else:
            st.info(
                "No processed data available. Please upload and process a PDF first."
            )

    with tab3:
        st.header("📋 Processing Logs and Progress")

        # Enhanced download logs section
        if st.session_state.processing_logs or st.session_state.batch_logs:
            st.subheader("📥 Download Options")
            col1, col2, col3 = st.columns(3)

            with col1:
                if st.session_state.processing_logs or st.session_state.get(
                        'all_logging_messages'):
                    # Comprehensive log file with ALL logging module data
                    comprehensive_logs = "=== COMPLETE PDF PROCESSING SESSION LOG ===\n"
                    comprehensive_logs += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                    comprehensive_logs += f"Total Batches: {st.session_state.total_batches}\n"
                    comprehensive_logs += f"Current Batch: {st.session_state.current_batch}\n"
                    comprehensive_logs += f"Processing Complete: {st.session_state.processing_complete}\n\n"

                    # Add ALL logging module messages first (most comprehensive)
                    if st.session_state.get('all_logging_messages'):
                        comprehensive_logs += "=== ALL LOGGING MODULE OUTPUT ===\n"
                        for log_entry in st.session_state.all_logging_messages:
                            timestamp = log_entry.get('timestamp', '')
                            level = log_entry.get('level', 'info').upper()
                            module = log_entry.get('module', 'unknown')
                            function = log_entry.get('function', 'unknown')
                            line = log_entry.get('line', 0)
                            message = log_entry.get('message', '')
                            comprehensive_logs += f"[{timestamp}] {level} - {module}.{function}:{line} - {message}\n"
                        comprehensive_logs += "\n"

                    # Add batch logs
                    if st.session_state.batch_logs:
                        comprehensive_logs += "=== BATCH PROGRESS LOGS ===\n"
                        for batch_log in st.session_state.batch_logs:
                            comprehensive_logs += f"{batch_log}\n"
                        comprehensive_logs += "\n"

                    # Add processing logs (UI level)
                    if st.session_state.processing_logs:
                        comprehensive_logs += "=== UI PROCESSING LOGS ===\n"
                        for log in st.session_state.processing_logs:
                            timestamp = log.get("timestamp", datetime.now())
                            if isinstance(timestamp, str):
                                timestamp = datetime.fromisoformat(
                                    timestamp.replace('Z', '+00:00'))
                            level = log.get("level", "info").upper()
                            message = log.get("message", "")
                            comprehensive_logs += f"[{timestamp.strftime('%Y-%m-%d %H:%M:%S')}] {level}: {message}\n"

                    st.download_button(
                        label="📄 Download ALL Logs",
                        data=comprehensive_logs,
                        file_name=
                        f"complete_logging_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain",
                        help=
                        "Download ALL logging module output plus session information"
                    )

            with col2:
                if st.session_state.batch_logs:
                    batch_logs_text = "=== BATCH PROGRESS LOG ===\n"
                    batch_logs_text += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                    for batch_log in st.session_state.batch_logs:
                        batch_logs_text += f"{batch_log}\n"

                    st.download_button(
                        label="📊 Download Batch Logs",
                        data=batch_logs_text,
                        file_name=
                        f"batch_progress_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain",
                        help="Download only batch processing progress logs")

            with col3:
                if st.session_state.processing_logs:
                    # Processing logs only (errors/warnings focus)
                    error_logs = [
                        log for log in st.session_state.processing_logs
                        if log.get("level") in ["error", "warning"]
                    ]
                    if error_logs:
                        error_logs_text = "=== ERROR AND WARNING LOGS ===\n"
                        error_logs_text += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                        for log in error_logs:
                            timestamp = log.get("timestamp", datetime.now())
                            if isinstance(timestamp, str):
                                timestamp = datetime.fromisoformat(
                                    timestamp.replace('Z', '+00:00'))
                            level = log.get("level", "").upper()
                            message = log.get("message", "")
                            error_logs_text += f"[{timestamp.strftime('%Y-%m-%d %H:%M:%S')}] {level}: {message}\n"

                        st.download_button(
                            label="⚠️ Download Errors Only",
                            data=error_logs_text,
                            file_name=
                            f"error_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain",
                            help="Download only error and warning messages")

        # Show batch progress if processing
        if st.session_state.total_batches > 0:
            display_batch_progress(st.session_state.current_batch,
                                   st.session_state.total_batches,
                                   st.session_state.batch_logs)

        # Show processing logs
        display_processing_logs(st.session_state.processing_logs)

    with tab4:
        st.header("🔍 JSON Data Viewer")

        if st.session_state.raw_json_data:
            # Display raw JSON with viewer
            display_json_viewer(st.session_state.raw_json_data,
                                "Raw Extracted Data")

            # Show individual records
            if st.checkbox("Show individual records"):
                record_index = st.slider(
                    "Select record:", 0,
                    len(st.session_state.raw_json_data) - 1, 0)
                selected_record = st.session_state.raw_json_data[record_index]
                display_json_viewer(selected_record,
                                    f"Record {record_index + 1}")
        else:
            st.info("No JSON data available. Please process a PDF first.")

    with tab5:
        st.header("📝 Prompts and Debug Information")

        if st.session_state.results_dir:
            display_prompts_viewer(st.session_state.results_dir)

            # Show chunk files
            st.subheader("📁 Generated Files")
            chunks_dir = st.session_state.results_dir / "chunks"
            if chunks_dir.exists():
                chunk_files = list(chunks_dir.glob("*.json"))
                if chunk_files:
                    selected_chunk = st.selectbox("Select chunk file:",
                                                  chunk_files,
                                                  format_func=lambda x: x.name)
                    if selected_chunk:
                        try:
                            with open(selected_chunk, 'r',
                                      encoding='utf-8') as f:
                                chunk_data = json.load(f)
                            display_json_viewer(
                                chunk_data, f"Chunk: {selected_chunk.name}")
                        except Exception as e:
                            st.error(f"Could not read chunk file: {e}")
        else:
            st.info(
                "No debug information available. Please process a PDF first.")


if __name__ == "__main__":
    main()
