# MEASURING HURTFUL SENTENCE COMPLETION IN FILMBERT MODELS USING HONEST

## Description of the project

The aim of this project is to evaluate and analyse bias transfer from the OpenSubtitles Corpus into pretrained models. 
For this purpose, I have fine-tuned with masked languaged modeling 3 pretained models available in Huggingface ([BERT-base-uncased](https://huggingface.co/bert-base-uncased), [RoBERTa-base](https://huggingface.co/roberta-base) and [distilBERt-base-uncased](https://huggingface.co/distilbert-base-uncased). 
For the fine-tuning I have used a small part of the OpenSubtitles Corpus in English; this corpus is available on the [OpenCorpus repository](https://opus.nlpl.eu/OpenSubtitles-v2018.php).
The models, after fine tuning, have been evaluated using the HONEST evaluation benchmark developed by Nozza et. al(2021) which is available in their [github](https://github.com/MilaNLProc/honest).

## Materials and methods
For this project first, I split the OpenSubtitle Corpus un .txt files of 1M lines each. Then I cleaned the corpus and transformed each of the files into a .csv file.
Since the corpus is very big only one of those files was split intro training, development and testing sets. This splited corpus was then transformed into a 
hugginface dataset in order to be able to use it as with the Hugginface API.

For the development of the code to the models,  have made use of the code available in the [Huggingface API tutorial](https://huggingface.co/docs/transformers/main/tasks/masked_language_modeling) as well as the [Hugginface Notebook on fine tuning with MLM](https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/language_modeling.ipynb#scrollTo=nFJ49iHJ3l_Z).
I have also used the code needed to run HONEST and evaluate the model I have created, which is available [here](https://colab.research.google.com/drive/13iAwHmtdYIAzDt8O5Ldat2vbKz9Ej6PT?usp=sharing).

All the materials used have been compiled and are available here in this github:

[LINK TO THE COLAB FOLDER](https://drive.google.com/drive/folders/1ialmqclR5IaD8573ijQ0Zj4tS5uO_Woq?usp=share_link)

1. Code used to preprocess the .txt files and covert them into .csv.
2. Code used to train and evaluate the models: [for models trained with 95.000 examples](https://github.com/AmaiaSolaun/Measuring-Hurtful-Sentence-Completion-In-Filmbert-Models-Using-Honest/blob/3fecb43e3c1608ccc57d458b9d11d27972201067/FINE_TUNING_WITH_FILM_DATA_IN_MLM_(95_000_examples).ipynb) and [for models trained with 20.000 examples](https://github.com/AmaiaSolaun/Measuring-Hurtful-Sentence-Completion-In-Filmbert-Models-Using-Honest/blob/0f692802e16bc816782be92cde349f7d2341eb3d/FINE_TUNING_WITH_FILM_DATA_IN_MLM_(20_000_examples).ipynb)
3. Trained FilmBERT models: [20000DistilFilmBERT-base-uncased](https://huggingface.co/AmaiaSolaun/film20000distilbert-base-uncased), [20000FilmBERT-base-uncased](AmaiaSolaun/film20000bert-base-uncased), [20000FilmRoBERTa-base](https://huggingface.co/AmaiaSolaun/film20000roberta-base),  [95000DistilFilmBERT-base-uncased](https://huggingface.co/AmaiaSolaun/film95000distilbert-base-uncased), [95000FilmBERT-base-uncased](AmaiaSolaun/film95000bert-base-uncased), [95000FilmRoBERTa-base](https://huggingface.co/AmaiaSolaun/film95000roberta-base).



## References

@inproceedings{nozza-etal-2021-honest,
    title = {"{HONEST}: Measuring Hurtful Sentence Completion in Language Models"},
    author = "Nozza, Debora and Bianchi, Federico  and Hovy, Dirk",
    booktitle = "Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jun,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.naacl-main.191",
    doi = "10.18653/v1/2021.naacl-main.191",
    pages = "2398--2406",
}

@inproceedings{nozza-etal-2022-measuring,
    title = {Measuring Harmful Sentence Completion in Language Models for LGBTQIA+ Individuals},
    author = "Nozza, Debora and Bianchi, Federico and Lauscher, Anne and Hovy, Dirk",
    booktitle = "Proceedings of the Second Workshop on Language Technology for Equality, Diversity and Inclusion",
    publisher = "Association for Computational Linguistics",
    year={2022}
}

@article{Sanh2019DistilBERTAD,
  title={DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter},
  author={Victor Sanh and Lysandre Debut and Julien Chaumond and Thomas Wolf},
  journal={ArXiv},
  year={2019},
  volume={abs/1910.01108}
}

@article{DBLP:journals/corr/abs-1810-04805,
  author    = {Jacob Devlin and
               Ming{-}Wei Chang and
               Kenton Lee and
               Kristina Toutanova},
  title     = {{BERT:} Pre-training of Deep Bidirectional Transformers for Language
               Understanding},
  journal   = {CoRR},
  volume    = {abs/1810.04805},
  year      = {2018},
  url       = {http://arxiv.org/abs/1810.04805},
  archivePrefix = {arXiv},
  eprint    = {1810.04805},
  timestamp = {Tue, 30 Oct 2018 20:39:56 +0100},
  biburl    = {https://dblp.org/rec/journals/corr/abs-1810-04805.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}


@article{DBLP:journals/corr/abs-1907-11692,
  author    = {Yinhan Liu and
               Myle Ott and
               Naman Goyal and
               Jingfei Du and
               Mandar Joshi and
               Danqi Chen and
               Omer Levy and
               Mike Lewis and
               Luke Zettlemoyer and
               Veselin Stoyanov},
  title     = {RoBERTa: {A} Robustly Optimized {BERT} Pretraining Approach},
  journal   = {CoRR},
  volume    = {abs/1907.11692},
  year      = {2019},
  url       = {http://arxiv.org/abs/1907.11692},
  archivePrefix = {arXiv},
  eprint    = {1907.11692},
  timestamp = {Thu, 01 Aug 2019 08:59:33 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-1907-11692.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
