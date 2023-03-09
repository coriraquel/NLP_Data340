[# Dataset Creation] ()
##### Due date
##### March 20th, 2023

## Milestone description
Create a dataset to perform your analysis. 

## Objective
This assignment focuses on the explaination of the dataset I plan to use for my final project investigating the creation, preparation, previous modelings and documentation of the dataset. 

## Sources of the dataset 
##### a. Where did you get the data? 
   My dataset game from Kaggle. I wanted to start with a simpler dataset to build my model around first before applying my own webscrapper later on in the project. My plan is to apply the new data onto the old model created with the simpler dataset then fix any descrprencies between the two. 
        
##### b. How did you get the data? 
   I downloaded this dataset off of the Kaggle website.
        
##### c. What is the license of the data if any? 
   This data is licensed under Attribution 4.0 International (CC BY 4.0) *(https://creativecommons.org/licenses/by/4.0/)* meaning that I am free to copy, redistribute, remix, transform, or build upon the dataset for all uses including commercial usage. In exchange for these rights under Attribution 4.0 International (CC BY 4.0) I must provide proper credit, link the license, indicate any changes made to the dataset ensure users know these changes are not endorsed by original manufacturers and not add on any legal or technological blockages that would prevent others from using the data as the license allows. 
##### e. Link to code used to create the dataset.
   https://huggingface.co/datasets/joangaes/depression
## Description of the dataset 
##### a. What is the size of the dataset? 
   The dataset is 2 columns by 27978 rows.
        
##### b. What is the format of the dataset? 
   The format of the data is a csv file. 
        
##### c. What is the structure of the dataset? 
   The data is structured by text and label. Where text is a collection of comments collect from individuals and the label is 0 or 1 based on if the comment was considered poisionous or not. 

## Data models and data structures 
##### a. What are the data models used in the dataset?
   This dataset is unstructured since as of right now it is only a collection of comments and their human determined labels however, I plan to use a relational and document data model. Eventually if I would like to predict most aaccurate summeries I would have to incorporate some sort of heirarchical or network data model. 
         
##### b. What are the data structures used in the dataset?
   Since this dataset is unstuctured there are not structures in it yet however each comment can be considered a document, documents can be broken down into setences, each sentence contains words that will be considered tokens and these tokens can be taken into consideration for N-gram testing to analyze and define 'poisonous' in terms of this dataset. 

## Outcomes
   This is my beginning dataset for my final project but ultimately I don't really like it. I would have at least preferred a dataset of scholarly articles related to mental health issues to start off with but I understand the idea of starting small. This dataset will change, it is just a matter when. I will be able to make a more informative decision after I complete an EDA and see what is possible however, my main concerns right now is that while the dataset seems to be in great condition, it looks a little thin. 


