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
    1. Get dataset from DatasetManager
    2. Serialize dataset to JSON for subprocess
    3. Wrap user code in execution template
    4. Execute via subprocess with uv run python -c
    5. Capture and return stdout/stderr
    """
    import asyncio
    import json
    
    try:
        # Step 1: Get dataset
        df = DatasetManager.get_dataset(dataset_name)
        
        # Step 2: Serialize dataset
        dataset_json = df.to_json(orient='records')
        
        # Step 3: Create execution template
        # Need to properly indent user code
        import textwrap
        indented_user_code = textwrap.indent(python_code, '    ')
        
        execution_code = f'''
import pandas as pd
import numpy as np
import plotly.express as px
import json

try:
    # Load dataset
    dataset_data = {dataset_json}
    df = pd.DataFrame(dataset_data)
    
    # Execute user code
{indented_user_code}
    
except Exception as e:
    print(f"ERROR: {{type(e).__name__}}: {{str(e)}}")
    import traceback
    print("Traceback:")
    print(traceback.format_exc())
'''
        
        # Step 4: Execute subprocess
        process = await asyncio.create_subprocess_exec(
            'uv', 'run', '--with', 'pandas', '--with', 'numpy', '--with', 'plotly',
            'python', '-c', execution_code,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT
        )
        
        # Step 5: Get output with timeout
        try:
            stdout, _ = await asyncio.wait_for(process.communicate(), timeout=30.0)
            return stdout.decode('utf-8')
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            return "TIMEOUT: Code execution exceeded 30 second limit"
            
    except Exception as e:
        return f"EXECUTION ERROR: {type(e).__name__}: {str(e)}"


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
