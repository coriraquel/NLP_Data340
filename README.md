# NLP_Data340
## The Problem:
The objective is to build a text summarization model to identify the most important sentences in a Harry Potter chapter and use those sentences to create extractive summaries of all chapters for each book. The summary would then go through a language generator as a starting point for creating a short story that expands on ideas presented in the summary. The language generator would utilize a bidirectional encoder to adjust the writing style to reflect more accurately with that of the summary 

## Learning Objective: 
1. Utilize advanced pandas techniques to efficiently process large amounts of text files 
2. Understand and implement basic NLP techniques to prep text files for further utilization 
3. Understand and implement LexRank 
4. Understand and implement BERT and GPT-2 models 

## Learner Growth: 
I believe this project is a good mix of natural language processing techniques in the area of machine learning and neural networks that became of interest to me throughout the semester. I believe adding in the Harry Potter aspects grounds the project in creativity and fun as well as allows me to better analyze the results as I am well versed in the initial topics. However, I believe this has many applications outside of just Harry Potter. 

## Potential Questions: 
1. Is there a pattern between chapters in the type of sentences the model is choosing as important before fine-tuning?
- Unfortunately, due to the small amount of data, the models trained off of the summaries were too incoherent to manually assess any significant patterns. 
2. What additional data would be useful in the creation of this project?
- More data, instead of doing this kind of project on one particular series looking back I would opt to do this on a genre such as YA dystopian novels or YA romance. This way you can continue adding book titles until the model has enough data to create meaningful conclusions. 
3. What machine learning techniques can be implemented to help the models perform better outside of their initial use? 
- I would utilize some database techniques in order to provide the model with a crutch in case there was not enough data to draw a meaningful conclusion. 
4. Can this code be updated to be used/applied to topics outside of how it was initially utilized? 
- Yes, I think this code can be utilized for a wide variety of literature from online blogs to academic texts. 

## Timeline: 
- Update: The project took 3 weeks for planning, data exploration, and research on the technique. The models were built over the course of a 5-week period with time varying between 3-10 hours a week. 

## Outcome: 
While I wouldn't go as far as to say I am disappointed in my model, it did not yield anything close to the desired outcome I hoped for. The main challenge faced was the limited amount of available data. Initially, I believed that the Harry Potter series would provide more than a sufficient amount of source material for basic generation and eventually fine-tuning the BERT model. However, I vastly underestimated the amount of information the BERT model can effectively process. In practice, the entire book series represents a fraction of the data BERT needs for a proper training data size. Even when using all the summaries up to the specified book and chapter as a reference, it was not enough for the model to effectively learn the series patterns. Due to the insufficient amount of data, I had challenges generating coherent summaries and text. Nevertheless, this project has given me proof of concept to believe that summarizing academic essays for students is achievable. To adapt it for practical use with younger age groups, for series such as Harry Potter, I would ideally construct a reference database that utilizes a rule-based system or a template-style approach to generate meaningful summaries. In the future, I am excited to explore additional machine-learning techniques to fine-tune the GPT-2 model. Although I considered running my text generator on the full Harry Potter texts, that project is well-researched and would deviate from the focus of my final. Overall, I have gained valuable insight working with BERT, LexRank, and GPT-2. Despite not achieving the desired success, I am pleased that I incorporated the stretch goal of sentiment analysis and gained insight into integrating sentiment models within the broader theme of my project as I believe it will be a valuable addition for the future. 
