import gradio as gr
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Load environment variables
import pandas as pd
df = pd.read_csv("./yenna_doc.csv")
scripts = df.Script.values
locations = df.Restaurant.values

load_dotenv()

# Initialize LLMs
llm = ChatOpenAI(temperature=0.8, max_tokens=3000, top_p=1, model="gpt-4o")
llm_claude = ChatAnthropic(model="claude-3-5-sonnet-20240620")

result_string = ""
for loc, script in zip(locations, scripts):
    result_string += f'"Restaurant name {loc}: Script {script}\n"'


# Define prompt templates and chains
extract_guidelines_template = """
Analyze the following restaurant review and extract key style elements:

{review}

Provide a detailed analysis covering:
1. Tone (e.g., enthusiastic, casual)
2. Language and word choices (e.g., informal, use of slang, simplicity of vocabulary)
3. Content structure (beginning, middle, end focus)
4. Descriptive language
5. Personal touch
6. Engagement techniques
7. Unique phrases or hashtags
8. Humor style
9. Use of emphasis (e.g., capitalization, punctuation)
10. Average word count of the entire script
11. Average sentence length
12. How the script begins
13. What the script focuses on in the middle
14. How the script ends
15. Simplicity of language (lack of high vocabulary words)
16. Any other notable stylistic elements

Format your response as a numbered list. For word count and sentence length, provide specific numbers.
"""

extract_guidelines_prompt = PromptTemplate(
    input_variables=["review"],
    template=extract_guidelines_template
)

extract_guidelines_chain = LLMChain(
    llm=llm,
    prompt=extract_guidelines_prompt,
    output_key="style_guidelines"
)

generate_review_template = """
Use the following style guidelines to create a new restaurant review:

```{style_guidelines}```

Restaurant Name: ```{restaurant_name}```
Cuisine Type: ```{cuisine_type}```
Location: ```{location}```
Special Feature: ```{special_feature}```

Dishes she ate:
```{Dishes}```

Additional notes for the script:
```{notes}```

Write a review that incorporates ALL the style elements mentioned above. Make sure to:
- Use the same word count and sentence length as specified
- Begin the script in the same style
- Focus on similar elements in the middle
- End the script in the same manner
- Use simple language throughout, avoiding high vocabulary words.
"""

generate_review_prompt = PromptTemplate(
    input_variables=["style_guidelines", "restaurant_name", "cuisine_type", "location", "special_feature", "Dishes", "notes"],
    template=generate_review_template
)

generate_review_chain = LLMChain(
    llm=llm_claude,
    prompt=generate_review_prompt,
    output_key="new_review"
)

overall_chain = SequentialChain(
    chains=[extract_guidelines_chain, generate_review_chain],
    input_variables=["review", "restaurant_name", "cuisine_type", "location", "special_feature", "Dishes", "notes"],
    output_variables=["style_guidelines", "new_review"],
    verbose=True
)

# Define the Gradio interface function
def generate_review(restaurant_name, cuisine_type, location, special_feature, dishes, notes):
    result = overall_chain({
        "review": result_string,
        "restaurant_name": restaurant_name,
        "cuisine_type": cuisine_type,
        "location": location,
        "special_feature": special_feature,
        "Dishes": dishes,
        "notes": notes
    })
    
    style_guidelines = result['style_guidelines']
    new_review = result['new_review']
    
    return style_guidelines, new_review

# Create the Gradio Interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("### Restaurant Script Generator")
    
    
    with gr.Row():
        restaurant_name = gr.Textbox(label="Restaurant Name", placeholder="Enter restaurant name")
        cuisine_type = gr.Textbox(label="Cuisine Type", placeholder="Enter cuisine type")
    
    with gr.Row():
        location = gr.Textbox(label="Location", placeholder="Enter restaurant location")
        special_feature = gr.Textbox(label="Special Feature", placeholder="Enter special feature (if any)")
    
    with gr.Row():
        dishes = gr.Textbox(label="Dishes", placeholder="Enter dishes the reviewer ate")
        notes = gr.Textbox(label="Additional Notes", placeholder="Enter any additional notes for the script")
    generate_btn = gr.Button("Generate Review")
    style_guidelines_output = gr.Textbox(label="Style Guidelines", placeholder="Style guidelines will appear here")
    new_review_output = gr.Textbox(label="New Restaurant Review", placeholder="Generated review will appear here")
    
    
    
    generate_btn.click(fn=generate_review, inputs=[restaurant_name, cuisine_type, location, special_feature, dishes, notes], outputs=[style_guidelines_output, new_review_output])

# Launch the Gradio UI
demo.launch()
