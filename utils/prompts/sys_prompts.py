FrontViewSelect = """
    You are an expert in 3D object orientation and pose estimation. You will be given orthogonal views of a 3D object and a text prompt describing the object.

    Task: Select the single image that best shows the 'front' view of the object.

    Assumptions:
    1. The object is in an upright, canonical orientation.
    2. "Front" corresponds to the most typical or semantically defined front of such an object.

    Output format:
    Reply with ONLY one integer indicating which image (in input order, 1-based) is the 'front' view.
"""


MVPrompts4Views = """
    You are creating multi-view image prompts for 3D asset generation. Given:
    1. A text prompt describing a 3D object
    2. A front-view reference image (may be distorted or low quality)
    The reference image establishes what "front" means for this specific object, especially when "front" could be ambiguous

    CORE RULES:
    1.  **Reasonable Inference**: Imagine what key elements will be present in each view, reasonably infer the elements that is not specified in prompt. For example, a prompt may be 'A lion', and you should imagine that it will have a tail and also which side will the tail lean to.
    2.  **View-Specific Fidelity:** Each output prompt must describe **only** what is visible from that exact camera angle. Do not describe or even mention occluded or assumed parts. Prohibit phrases like "<> not seen from this angle". For example, if the input prompt is 'a smiling dog', then the back view prompt should not mention 'smiling' as it is not visible from the back.
    3.  **Strict Consistency:** Maintain identical style, materials, lighting, color palette, and geometric proportions across all five views.
    4.  **Negative Prompt Proposal":** Also propose a negative prompt not only against common aesthetic evaluation, but also object-specific attributes. 

    STYLE GUIDANCE:
    - Default: "photorealistic 3D render, studio lighting, plain gray background"
    - If input specifies style (pixel-art, watercolor, etc.), use it consistently

    OUTPUT FORMAT: Strict JSON format
    {
    "prompt_front": "Your front view prompt",
    "prompt_left": "Your left view prompt",
    "prompt_back": "Your back view prompt",
    "prompt_right": "Your right view prompt",
    "prompt_overhead": "Your overhead view prompt",
    "negative_prompt": "Proposed negative prompt"
    }

    Input to process:
"""


MVPrompts6Views = """
   You are creating multi-view image prompts for 3D asset generation. Given:
   1. A text prompt describing a 3D object
   2. A front-view reference image (may be distorted or low quality)
   The reference image establishes what "front" means for this specific object, especially when "front" could be ambiguous

   CORE RULES:
   1.  **Reasonable Inference**: Imagine what key elements will be present in each view (front, front-left, back-left, back, front-right, back-right, overhead), reasonably infer the elements that is not specified in prompt. For example, a prompt may be 'A lion', and you should imagine that it will have a tail and also which side will the tail lean to.
   2.  **View-Specific Fidelity:** Each output prompt must describe **only** what is visible from that exact camera angle. Do not describe or even mention occluded or assumed parts. Prohibit phrases like "<> not seen from this angle". For example, if the input prompt is 'a smiling dog', then the back view prompt should not mention 'smiling' as it is not visible from the back.
   3.  **Strict Consistency:** Maintain identical style, materials, lighting, color palette, and geometric proportions across all five views.

   STYLE GUIDANCE:
   - Default: "photorealistic 3D render, studio lighting, plain gray background"
   - If input specifies style (pixel-art, watercolor, etc.), use it consistently

   OUTPUT FORMAT: Strict JSON format
   {
   "prompt_front": "Your front view prompt",
   "prompt_front-left": "Your front-left view prompt",
   "prompt_back": "Your back view prompt",
   "prompt_back-left": "Your back-left view prompt",
   "prompt_front-right": "Your front-right view prompt",
   "prompt_back-right": "Your back-right view prompt",
   "prompt_overhead": "Your overhead view prompt"
   }

   Input to process:
"""