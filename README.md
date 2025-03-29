# Indian Caste Bias in AI: Embeddings, Hiring, and Ethical Gaps

Here is the code to replace the plots as seen in: \<fixed-paper-link goes here\>

## 1. Getting Started

Set up the environment using requirements.txt.

```
pip install -r requirements.txt
```

(Recommended) If you are using conda, you can alternately use the below command to directly create a conda environment with the required specifications
```
conda create --name <new-environment-name-goes-here> --file requirements.txt
```

## 2. Getting Caste Bias WEAT Scores in Embeddings

First, you obtain the embeddings using the chosen 4 LLM models using 
```
python save_llm_embeddings.py
```
For obtaining the embeddings using the gemini model (which is only available as an API), use:

```
python save_gemini_embeddings.py
```
You might need to use the above script multiple times due to API time-out issues. This can be automated using a shell-script that executes this script periodically with time-gaps for a set duration of time. At least, that is what I did. 
The embeddings are saved in the the `data/` subfolder as JSON files. 

Once you have these set of embeddings, you can obtain the WEAT score results using:

```
python get_weat_scores.py
```
## 3. Examining Caste-Bias in LLM used for Hiring

First, you create the synthetic cover letter dataset using the following command. This dataset is obtained by using https://huggingface.co/datasets/ShashiVish/cover-letter-dataset as a starting point!

```
python create_job_dataset.py
```
Once this is done, you can obtain the LLM decisions, which uses the GPT-4o model, written into the `data/` subfolder, by using:
```
python get_LLM_hiring_results.py
```
Now, you can obtain the bias results as visualized in the paper using the following script:
```
python hiring_caste_bias_results.py
```





 
