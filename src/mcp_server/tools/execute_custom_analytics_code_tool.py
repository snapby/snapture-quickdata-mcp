"""Custom analytics code execution tool implementation."""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from ..models.schemas import DatasetManager, loaded_datasets, dataset_schemas
from ..advanced.advanced_code_executor import AdvancedCodeExecutor


async def execute_custom_analytics_code(dataset_name: str, python_code: str) -> str:
    """
    Execute custom Python code against a loaded dataset.
    
    Implementation steps:
    1. Get dataset from DatasetManager.
    2. Create a temporary file and serialize the dataset to Parquet format for efficiency.
    3. Pass the path to the temporary file as an argument to the subprocess.
    4. The subprocess script reads the DataFrame from the Parquet file.
    5. Wrap user code in an execution template.
    6. Execute via subprocess with `uv run python -c`.
    7. Capture and reDataFrameturn stdout/stderr, ensuring the temporary file is cleaned up.
    """
    import asyncio
    import tempfile
    import os
    import textwrap

    temp_file_path = None
    try:
        # Step 1: Get dataset
        df = DatasetManager.get_dataset(dataset_name)

        # Step 2: Create a temporary file and serialize the dataset to Parquet
        with tempfile.NamedTemporaryFile(delete=False, suffix=".parquet") as temp_file_context:
            temp_file_path = temp_file_context.name
        df.to_parquet(temp_file_path, engine='pyarrow')

        # Step 3: Create execution template
        indented_user_code = textwrap.indent(python_code, '    ')
        
        execution_code = f'''
import pandas as pd
import numpy as np
import plotly.express as px
import sys

try:
    # Step 4: Load dataset from Parquet file path passed as an argument
    dataset_path = sys.argv[1]
    df = pd.read_parquet(dataset_path)
    
    # Execute user code
{indented_user_code}
    
except Exception as e:
    print(f"ERROR: {{type(e).__name__}}: {{str(e)}}")
    import traceback
    print("Traceback:")
    print(traceback.format_exc())
'''
        
        # Step 6: Execute subprocess, passing the temp file path as an argument
        process = await asyncio.create_subprocess_exec(
            'uv', 'run', '--with', 'pandas', '--with', 'numpy', '--with', 'plotly', '--with', 'pyarrow',
            'python', '-c', execution_code, temp_file_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT
        )
        
        # Step 7: Get output with timeout
        try:
            stdout, _ = await asyncio.wait_for(process.communicate(), timeout=30.0)
            return stdout.decode('utf-8')
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            return "TIMEOUT: Code execution exceeded 30 second limit"
            
    except Exception as e:
        return f"EXECUTION ERROR: {type(e).__name__}: {str(e)}"
    finally:
        # Step 8: Ensure temporary file is cleaned up
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)


async def execute_enhanced_analytics_code(
    dataset_name: str, 
    python_code: str,
    execution_mode: str = "safe",
    include_ai_context: bool = True,
    timeout_seconds: int = 30
) -> dict:
    """
    Enhanced custom Python code execution with AI context and safety features.
    
    This function provides:
    - Automatic context injection with dataset schema
    - Code safety analysis and suggestions
    - Performance monitoring and optimization hints
    - Execution history tracking
    - AI-generated helper functions
    - Template suggestions for common patterns
    
    Args:
        dataset_name: Name of the loaded dataset to analyze
        python_code: Python code to execute
        execution_mode: Execution safety mode ('safe', 'standard', 'advanced')
        include_ai_context: Whether to include AI helper functions
        timeout_seconds: Maximum execution time in seconds
        
    Returns:
        dict: Enhanced execution result with insights and suggestions
    """
    executor = AdvancedCodeExecutor()
    return await executor.execute_enhanced_analytics_code(
        dataset_name=dataset_name,
        python_code=python_code,
        execution_mode=execution_mode,
        include_ai_context=include_ai_context,
        timeout_seconds=timeout_seconds
    )
