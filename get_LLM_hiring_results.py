import pandas as pd
import tiktoken
import openai  # OpenAI Python library
import asyncio
import aiohttp
import json
import re
from tqdm import tqdm  # Import tqdm for the progress bar
from config import OPENAI_API_KEY as API_KEY


model = 'gpt-4o'

client = openai.OpenAI(api_key=API_KEY)

# Load the dataset
df = pd.read_csv('new_job_dataset.csv')

def count_tokens(text, model):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def estimate_cost(model):
    df = pd.read_csv("new_job_dataset.csv")

    input_tokens = sum(count_tokens(str(x), model) for x in df["Prompt full"])

    sample_output = "[[4]] \
    Reasoning: The candidate meets most of the required qualifications, including over 3 years of experience in building machine learning models, \
    expertise in generative AI, and hands-on experience with Python and deep learning libraries. They also have experience in image/video generation, \
    which is relevant to the position. Their strong interpersonal skills and experience in collaborating with product and design teams are valuable assets. \
    However, the academic performance (CGPA of 6.77) might not be as strong as other candidates, slightly lowering their rating. \
    Overall, they are well-qualified but not a perfect fit."

    output_tokens = count_tokens(sample_output, model) * len(df)


    pricing_4o_mini = [0.15, 0.6]
    pricing_4o = [2.5, 10]

    print(f"Expected number of Input tokens: {input_tokens}, Output tokens: {output_tokens}, Total tokens: {input_tokens + output_tokens}")

    if model == 'gpt-4o-mini':
        print(f"Total Expected Cost (Per Iteration): {(input_tokens*pricing_4o_mini[0] + output_tokens*pricing_4o_mini[1])/1000000}")
    else:
        print(f"Total Expected Cost (Per Iteration): {(input_tokens*pricing_4o[0] + output_tokens*pricing_4o[1])/1000000}")

def extract_rating(response_text):
    match = re.search(r'\[\[(\d)\]\]', response_text)
    return match.group(1) if match else None

async def fetch_response(prompt):
    try:
        # Using the new Chat API
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.5
        )

        # Accessing the assistant's response
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error during API request: {e}")
        return "Error: Request failed."


# Function to send batched requests
async def process_batch(prompts):
    tasks = [fetch_response(prompt) for prompt in prompts]
    responses = await asyncio.gather(*tasks)
    return responses

async def process_data(df):
    all_responses = []
    batch_size = 10  # Number of prompts per batch

    # Use tqdm for progress bar
    for i in tqdm(range(0, len(df), batch_size), desc="Processing Batches"):
        batch = df['Prompt full'][i:i+batch_size].tolist()  # Create a batch
        responses = await process_batch(batch)
        
        for idx, response in enumerate(responses):
            rating = extract_rating(response)
            # Add the original row along with the response and rating
            all_responses.append({
                **df.iloc[i + idx].to_dict(),  # This adds all columns from the CSV row
                'Response': response.strip(),
                'Rating': rating
            })
    
    return all_responses

# Main execution flow
def main():
    print(f"Using model {model}....")
    estimate_cost(model)

    # Execute the async processing
    loop = asyncio.get_event_loop()
    all_responses = loop.run_until_complete(process_data(df))

    # Save the results to an output file (outputs.json)
    with open(f'data/results_{model}_full.json', 'w') as f:
        json.dump(all_responses, f, indent=4)

if __name__ == '__main__':
    main()
