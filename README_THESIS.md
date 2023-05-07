## Thesis guide 

Here we are using output from the thesis_specter repository. The output will be 
- bert-base-finnish-cased-v1 embeddings 
- sbert-cased-finnish-paraphrase embeddings
- Finetuned model embeddings. The model is finetuned with Wikipedia data starting from bert-base-finnish-cased-v1 

## Hypothesis 
- I want to test whether 
1. The SPECTER Framework works for finetuning another model from another domain 
2. The finetuned model obtains more accurate embeddings in two different task 
- Ranking direct cross-links 
- Classifying wiki classes 



## Initial
Run scripts 1-3

## Inference and evaluation
- Run
```
./3-predict.sh --thesis 
```
If you run without ***--thesis*** then original scidocs will be run.


### Metadata schema 

- This metadata can be used 

```json
{
  "<paper_id>": {
    "abstract": "<string>",
    "authors": [
      "<string>"
    ],
    "cited_by": [
      "<cite_paper_id>"
    ],
    "paper_id": "<paper_id>",
    "references": [
      "<outgoing_1>",
      "<outgoing_2>"
    ],
    "title": "Learning to Discriminate Noises for Incorporating External Information in Neural Machine Translation",
    "year": 2018
  }
}

```

```csv
pid,class_label
<paper_id, str>, <class, int>
<paper_id_2, str>, <class, int>
<paper_id_3, str>, <class, int>
```



### Direct citations:
* Around 30k total papers from held out pool of papers 
* 1000 papers and for each paper 5 cited papers and 25 randomly selected papers => 1000 * 30 = 30K

TODO: Find 1000 papers which have at least 5 cited papers from x different classes. 

### Co citations
- This could be done with same code. 
