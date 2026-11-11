# Exercise 7: Python APIs

## Project context

This repository is the solution to Assignment 7 - *Computing for Data Science*. 
1. The data used was found on https://www.openml.org and its gathered from participanst in experimental speed dating.
2. A machine learning model is trained once on this data and saved with `joblib`. 
3. A FastAPI server loads the saved model and returns predictions. 
4. A client script sends one example datapoint as JSON and prints the prediction it gets back.

## Model inputs 
model_inputs.json

They must be a rate between 0 and 10

    like: How much I liked my partner
    attractive_partner: I rate my partner: attractiveness
    funny_partner: I rate my partner: being funny
    shared_interests_partner: I rate my partner: shared interests
    guess_prob_liked: How likely I think my partner likes me
    attractive_o: My partner rates me: attractiveness
    funny_o: My partner rates me: being funny


## Data

`data/speeddating.arff` comes from speed dating experiments run at Columbia Business School between 2002 and 2004 (Fisman & Iyengar). Each participant had a four-minute "first date" with every participant of the opposite sex. They then said whether they wanted to see that person again and rated them on six attributes: attractiveness, sincerity, intelligence, fun, ambition and shared interests. The dataset also includes questionnaire answers on demographics, dating preferences, self-ratings and hobbies.

- **Rows:** 8,378 dates, each seen from one participant about one partner
- **Target:** `match`, which is 1 when both people said yes (about 16% of dates)
- **Features used:** seven 1–10 ratings from the date night, from both sides of the date: `like`, `attractive_partner`, `funny_partner`, `shared_interests_partner`, `guess_prob_liked`, `attractive_o`, `funny_o`

Reference: Fisman, R., Iyengar, S., Kamenica, E. & Simonson, I. (2006). Gender Differences in Mate Selection: Evidence From a Speed Dating Experiment. *The Quarterly Journal of Economics*, 121(2), 673–697. https://doi.org/10.1162/qjec.2006.121.2.673
