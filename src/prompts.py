reasoning_prompts = {
    "sam3_multi_expression": """
    Context: User Expression Q: '{Q}'. 
    Target Description: '{T}'.
    Task: Look at the image. Generate 3 distinct noun phrases that refer to EXACTLY THE SAME INSTANCE.
    CRITICAL CONSTRAINT:
    - The new phrase must still be a valid answer of Q.
    - The object name should be one in '{N}'.
    - KEEP the attribute and location description, do NOT miss information.
    - If the target is a part of one thing, use "xxx of xxx" to represent (e.g., 'ears of the dog').
    - If there are multiple similar instances, MUST use attribute and location of the target to distinguish.
    - Length: 2 to 8 words each phrase.
    Output 3 phrases separated by commas:""",

    "ref_refine": """Context: User Expression Q: '{Q}'. 
    Current Descriptions: '{TS}'.
    Task: Look at the image. Generate a **new noun phrase** that refer to **EXACTLY THE SAME TARGET** as Q.
    CRITICAL CONSTRAINT:
    - The ONLY object name of the phase should be one in '{N}'.
    - The new phase should not appear in current descriptions.
    - Start with "The" or "A".
    - Length: 2 to 5 words.
    Output only the new phase:""",
}
