# Evaluation Issues and Fixes

## Issues Found

### 1. JSON Truncation Error
**Problem**: The local Mistral model was truncating JSON responses during memory distillation, causing parse failures.

**Error Pattern**:
```
[distiller] Failed to parse JSON: {
  "memories": [
    {
      "action": "keep",
      "type": "preference",
      ...
```

**Root Cause**: The `max_tokens=1000` limit was too low for larger memory sets, especially with local models that may need more tokens to complete JSON structures.

**Fixes Applied**:
1. **Increased max_tokens**: Changed from 1000 to 2000 in `src/distiller.py`
2. **Added truncation recovery**: New logic detects and attempts to repair truncated JSON by:
   - Detecting `...` endings
   - Trying common closing bracket combinations
   - Logging warnings when recovery is successful
3. **Improved error handling**: Better logging to identify truncation vs other JSON errors

### 2. Response Format Compatibility
**Problem**: The `response_format={"type": "json_object"}` parameter is OpenAI-specific and may not work with all providers.

**Fix**: Added conditional response format based on client type.

## Current Status

The evaluation is running successfully with these fixes. The system now:
- Handles truncated responses gracefully
- Logs warnings instead of failing completely
- Continues evaluation even if some distillation attempts fail
- Provides comprehensive statistics at the end

## Testing the Fixes

To test with your local model:
```bash
# Watch for warnings instead of errors
uv run eval/evaluate.py --local --model mistral --turns 50

# Export results for analysis
uv run eval/evaluate.py --local --model mistral --turns 50 --export results.json
```

## Recommendations

1. **For local models**: Use models with larger context windows (8k+) to reduce truncation risk
2. **Monitor the logs**: Check `eval.log` for truncation warnings
3. **Adjust max_tokens**: If you still see truncation, increase beyond 2000 in `src/distiller.py`
4. **Compare providers**: Run the same evaluation with Groq to establish a baseline
