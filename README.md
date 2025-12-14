# League of Legends Early Objectives, Gold, and Win Conversion Statistical Analysis

League of Legends Early Objectives, Gold, and Win Conversion Statistical Analysis is a comprehensive data science project conducted at UCSD. The project encompasses various stages of analysis, starting from exploratory data analysis to hypothesis testing, creation and improvement of baseline models, and concluding with fairness analysis. The primary focus of the project is to investigate the significance of the objectives and early game advantages in League of Legends matches and their impact on match statistics and outcomes.

Author: Steven Ngo

## Introduction
### General Background
League of Legends (LOL) is a popular multiplayer online battle arena (MOBA) game developed by Riot Games. With millions of players worldwide, it has become one of the most influential and widely played esports in the gaming industry. The data set we will be working with is a professional data set that’s developed by Oracle's Elixir. The file records match data from professional LOL esports gaming matches throughout 2022. 

This dataset captures key gameplay statistics and outcomes from a collection of LOL matches, offering a rich source of information for understanding player behavior, team dynamics, and match outcomes. It includes a variety of features such as individual player performance, team strategies, in-game statistics, and overall match dynamics.

In League of Legends, the objectives hold significant weight and serve as an important factor that determines the outcome of a match. Objectives refer to the in-game quests available throughout the game, in which destroying or securing that objective grants a buff to the team's stats. Objectives typically require multiple players from opposing teams to compete against each other for this buff. These include the drakes (dragons), rift herald, towers, and baron. 

Beyond its immediate impact on the team's strength, securing these objectives *first* signifies degrees of map dominance, lane control, and team-fight strategic advantage. I am particularly interested in how securing some of these objectives in the early game affects match outcomes, along with other early team statistics, such as kills and gold, which are quantitative measures of a team's ability to dismantle their opponents.

The central question I am interested in is: **Do teams with early gold leads (at 15 minutes) convert to wins more reliably than teams with early objective leads**? In this project, I utilized data analysis techniques to quantify the impact of early gold and objectives on gaming statistics,  including individual team performance in-game metrics, and ultimately, match outcomes. Seeing how different features play into shaping this quest, I eventually explore their patterns through machine learning. My predictive model captures the real patterns behind professional play and how focusing on the early phases may be an overlooked driver of who wins and who loses in these pivotal games.

### Introduction of Columns
The dataset introduces a comprehensive array of columns featuring gameplay metrics and match outcomes from professional League of Legends esports matches. There are 148992 rows in this dataset, and here's an introduction to some of the key columns:

In the dataset provided, we encounter various columns that encapsulate essential gameplay statistics and match outcomes from professional League of Legends (LoL) esports matches. Here's a brief introduction to each of these columns:

The dataset introduces a comprehensive array of columns featuring gameplay metrics and match outcomes from professional League of Legends esports matches. Here's an introduction to the key columns used in this analysis:

- `gameid`: This column represents a unique identifier for each individual match played. It allows us to distinguish between different matches in the dataset.

- `result`: This column indicates the outcome of a match for a specific team or player. 1 indicates the team or the team that the player is in won, 0 indicates lost.

- `kills`: The 'kills' column quantifies the number of enemy champions a player or team successfully eliminated during the match.

- `deaths`: Conversely, the 'deaths' column records the number of times a player or team was eliminated by enemy champions.

- `assists`: The 'assists' column records the number of assists credited to a player or team, indicating instances where they contributed to eliminating an enemy champion without securing the kill themselves.

- `position`: The 'position' column specifies the role or position played by an individual player within their team composition. Common positions include 'top,' 'jungle,' 'mid,' 'bot,' and 'support.'

- `firstdragon`: This binary column indicates whether a team secured the first dragon of the match (1 for yes, 0 for no). First dragon provides an early advantage through elemental buffs.

- `firstherald`: This binary column denotes whether a team secured the first Rift Herald (1 for yes, 0 for no), which can be used to gain turret plating advantages and early map pressure.

- `firstbaron`: This binary column indicates whether a team secured the first Baron Nashor of the match (1 for yes, 0 for no). Baron provides significant team-wide buffs and is often a game-deciding objective.

- `firsttower`: This binary column indicates whether a team destroyed the first tower of the match (1 for yes, 0 for no). The first tower provides additional gold and opens up map control opportunities.

- `damageshare`: This column represents the percentage of total team damage dealt by an individual player. It indicates how much of the team's damage output is contributed by that player, showing their impact in team fights.

- `goldat15`: This column records the total gold accumulated by a player or team at the 15-minute mark of the game, a key benchmark for assessing early game performance and advantage.

- `teamkills`: This column records the total number of kills secured by the entire team during the match, reflecting the team's overall aggression and combat success.

- `gamelength`: This column indicates the total duration of the match in seconds (or minutes), showing how long the game lasted from start to finish.

- `earnedgold`: This column tracks the amount of gold earned by a player or team through kills, objectives, and other active gameplay, excluding passive gold generation.

- `totalgold`: This column represents the total gold accumulated by a player or team throughout the entire match, including both earned gold and passive gold generation.

- `teamkpm`: This column calculates the team's kills per minute, providing a rate statistic that measures how aggressively and frequently a team secures kills relative to game duration.

## Data Cleaning and Exploratory Data Analysis
### Data Cleaning
To save time in the further data cleaning steps, I first only keep the relevant columns: `result`, `position`, `firstdragon`, `firstherald`, `firstbaron`, `firsttower`, `damageshare`, `goldat15`, `teamkills`, `gamelength`, `earnedgold`, `totalgold`, `teamkpm`. In this dataset, each game has 12 rows, with 10 rows representing each of the players (i.e. player rows), and 2 rows immediately following that summarize the overall team performance and result. I decided to keep the team rows, as I am focusing on the comparisons in statistics between teams, rather than players, in this project.

Furthermore, among these columns, we find out that the columns `firstdragon`, `firstherald`, `firstbaron`, `firsttower` have some missing values. For these `first...` categorical variables, filling with 0 makes perfect sense because:

0 = False/No - neither team secured first dragon/herald/tower
1 = True/Yes - the team secured first dragon/herald/tower

This is a meaningful categorical value, not a missing data issue. In League of Legends, it's possible that:

No team gets first dragon (if the game ends quickly or teams ignore dragons)
No team gets first herald (similar reasoning)
First tower could theoretically not fall in very unusual circumstances

Below is the head of my team_df_filtered dataframe.

| index | result | teamkills | deaths | assists | ... | gamelength | earnedgold | totalgold | teamkpm |
|-------|--------|-----------|--------|---------|-----|------------|------------|-----------|---------|
| 10    | 0      | 9         | 19     | 19      | ... | 1713       | 28222.0    | 47070     | 0.32    |
| 11    | 1      | 19        | 9      | 62      | ... | 1713       | 33769.0    | 52617     | 0.67    |
| 22    | 0      | 3         | 16     | 7       | ... | 2114       | 34688.0    | 57629     | 0.09    |
| 23    | 1      | 16        | 3      | 39      | ... | 2114       | 48063.0    | 71004     | 0.45    |
| 34    | 1      | 13        | 6      | 35      | ... | 1365       | 30167.0    | 45468     | 0.57    |

The cleaned dataset here consists of all the columns I need for **exploratory data analysis**, **hypothesis testing**, and **prediction model**. As we traverse through the different sections, I'll be transparent about what columns that I am focusing on for specific sections (as there are many objectives and features to keep track of).

### Univariate Analysis
I performed univariate analysis on the kill and damage share statistics in the dataset.

<iframe
  src="assets/fig1.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

This histogram displays the distribution of total team kills across all matches in the dataset. The distribution appears approximately normal with a slight right skew, centered around 15-17 kills per team. The majority of games see teams securing between 10 and 25 kills, with the peak probability occurring around 16-18 kills. There are very few games with extremely high kill counts (above 35 kills), suggesting that professional matches tend to be relatively controlled in terms of combat frequency.

### Bivariate Analysis

I performed some bivariate analyses on the relationship between securing first tower and first dragon and match outcomes.

<iframe
  src="assets/fig5.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

This grouped bar chart compares win rates based on whether teams secured the first tower. Teams that destroyed the first tower won approximately 62% of their games (light blue), while teams that did not secure first tower won only about 38%. This stark difference demonstrates the strategic importance of securing the first tower objective. It provides both an immediate gold advantage and opens up map control opportunities that cascade into late-game advantages.

<iframe
  src="assets/fig7.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

This pie chart illustrates the win rate distribution among teams that secured the first dragon objective. While securing first dragon provides a notable edge, the relatively close split suggests that first dragon alone is not determinative of match outcomes; it indicates that while early dragon control is valuable, teams can still overcome this deficit through superior macro play and teamfighting.

### Interesting Aggregates

| firstherald | teamkills | assists | deaths | gamelength | earnedgold | totalgold | teamkpm | goldat15  |
|-------------|-----------|---------|--------|------------|------------|-----------|---------|-----------|
| False       | 13.68     | 30.80   | 15.18  | 1889.57    | 35289.83   | 55934.23  | 0.44    | 24240.08  |
| True        | 15.60     | 34.42   | 13.63  | 1901.84    | 37476.02   | 58252.79  | 0.51    | 25413.94  |

This aggregated data shows that teams securing the first Rift Herald demonstrate consistently better performance across all metrics. Teams with first herald average 1.92 more kills (15.60 vs 13.68), maintain better KDAs with fewer deaths (13.63 vs 15.18) and more assists (34.42 vs 30.80), and earn approximately 1,186 more gold (37,476 vs 35,290). They also achieve higher kills per minute (0.51 vs 0.44) and accumulate more gold by the 15-minute mark (25,414 vs 24,240), indicating stronger early game control. Interestingly, games where teams secure first herald tend to last slightly longer (1,901 vs 1,889 seconds), suggesting these teams may play more methodically to secure their advantages. This data highlights the strategic importance of Rift Herald in establishing early-to-mid game dominance.

## Assessment of Missingness

### NMAR Analysis

In our data, I believe the columns `ban1`, `ban2`, `ban3`, `ban4`, and `ban5` are all Not Missing At Random (NMAR) because they're missingness depends on the value itself. In the game, particularly the draft and banning phase, players can either choose a champion to ban from being picked, or choose to ban nothing. The player might choose to not ban due to internet connection issues, personal preferences, being lazy to ban a champion, missing the opportunity to ban after the timer runs out, or other factors; all of which result in an NMAR missingness mechanism.

### Missingness Dependency

For an assessment relevant to early game objectives/stats, I chose to analyze the missingness of the original `goldat15` column against `gamelength` and `result`, assessing whether or not `goldat15` is Missing At Random (MAR) or (MCAR) dependent on these two columns. The significance level I chose for both permutation tests is 0.05, and the test statistic is the absolute difference of means.

First, I perform the permutation test on `goldat15` and `gamelength`, and the missingness of `goldat15` **does** depend on `gamelength`. 

**Null Hypothesis**: Distribution of `gamelength` when `goldat15` is missing is the same as the distribution of `gamelength` when `goldat15` is not missing.

**Alternative Hypothesis**: Distribution of `gamelength` when `goldat15` is missing is NOT same as the distribution of `gamelength` when `goldat15` is not missing.

Below is the observed distribution of `gamelength` when `goldat15` is missing and not missing, in both a summary table and under a KDE plot.

|                          | count | mean    | median | std    | min | max  |
|--------------------------|-------|---------|--------|--------|-----|------|
| goldat15_missing=False   | 21312 | 1901.90 | 1864.0 | 339.92 | 933 | 3577 |
| goldat15_missing=True    | 3786  | 1854.67 | 1824.0 | 329.42 | 774 | 3363 |

<iframe
  src="assets/fig9.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

After performing the permutation tests, I found that the **observed statistic** for this permutation test is: 47.2313, and the **p-value** is 0.0000. The plot below shows the empirical distribution of the absolute difference in average game lengths for the test.

<iframe
  src="assets/fig11.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

Since the p-value is less than the 0.5 significance level, we reject the null hypothesis. Thus, the missingness of `goldat15` depends on the `gamelength` column. 

The second permutation test that I perfomed is on `goldat15` and `result`, and the missingness of `goldat15` does not depend on `result`. 

**Null Hypothesis**: Distribution of `result` when `goldat15` is missing is the same as the distribution of `result` when `goldat15` is not missing.

**Alternative Hypothesis**: Distribution of `result` when `goldat15` is missing is NOT same as the distribution of `result` when `goldat15` is not missing.

Below is the observed distribution of `result` when `goldat15` is missing and not missing as a table and a KDE curve.

| result | goldat15_missing = False | goldat15_missing = True |
|--------|--------------------------|-------------------------|
| 0      | 0.5                      | 0.5                     |
| 1      | 0.5                      | 0.5                     |

<iframe
  src="assets/fig10.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

After I performed permutation tests, I found that the **observed statistic** for this permutation test is: 0.0000, and the **p-value** is 1. The plot below shows the empirical distribution of the absolute differences in win proportions for the test.

<iframe
  src="assets/fig12.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

Since the p-value is greater than the 0.5 significance level, we fail to reject the null hypothesis. Thus, the missingness of `goldat15` does not depend on the `result` column. 

## Hypothesis Testing
In the hypothesis test, I aimed to assess whether there is a significant difference in the distribution of proportions of winning games between the team that secures the first dragon and the team that does not. During EDA, we had a closer look at how first dragon-securing teams had a higher winrate than those that did not. This is an opportunity to evaluate a team's ability to convert their first dragon advantage into a quantitative measure of dominance, like total kills at the end of the match.

**Null**: The distribution of kills for the team that secures the first dragon is the **same** as the team that does not get the first dragon. 

**Alternative**: The distribution of kills for the team that secures the first dragon is **greater** as the team that does not get the first dragon.

**Test Statistic**: *Signed* mean difference between teams in kills with and without first kills.

**Significance Level**: 5%

Here is a histogram containing the distribution of our test statistics during the hypothesis test:

<iframe
  src="assets/fig13.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

Based on the hypothesis test performed, with a **p-value** of **0.0000**, we **reject** the null hypothesis. This suggests that the distribution of kills for winning games for the team that secures the first dragon is greater than the team that does not get the first dragon. This finding shows that first dragon is likely to have a non-neglegible an impact on a team's ability to dominate the game through kills.

## Framing a Prediction Problem
From the last section, we found out that getting first objectives may have a significant impact on the team kills. Since the statistics for each team could be so different, are there any specific characteristics for each position in terms of in-game statistics? In other words, is it possible to predict a player's position solely by analyzing their in-game early statistics, such as first dragon, first baron, first tower, and early quantities?

To address this question, I can employ machine learning techniques such as classification algorithms. For my prediction model, I will focus on the result of the game for a given team. Thus, my **prediction problem**: *Are we able to predict the outcome of a game based on a team's early game statistics?* My model seeks to answer this question.

In this part, we will need to one-hot encode the original `firstdragon`, `firstherald`, `firstbaron`, `firsttower` columns, and this will give me 4 binary columns representing that team's state. The `goldat15` and `killsat15` columns will need to be scaled for interpretability of feature importances and Thus, this is a `binary classfication model`, and my target variable is `result`. 

| index | firstdragon | firstherald | firsttower | firstbaron | goldat15_imputed | killsat15_imputed |
|-------|-------------|-------------|------------|------------|------------------|-------------------|
| 10    | False       | True        | True       | False      | 24730.30         | 4.73              |
| 11    | True        | False       | False      | False      | 27644.66         | 9.98              |
| 22    | False       | True        | False      | False      | 24534.58         | 1.28              |
| 23    | True        | False       | True       | True       | 30228.76         | 6.81              |
| 34    | False       | False       | True       | False      | 29978.90         | 8.57              |

To prevent overfitting, the data will be partitioned into two splits: 80% training data, and 20% test data. In terms of model evaluation, I will focus on model accuracy. Since the data that I am working with is already balanced, there is no need to use other metrics like F1, Precision, or Recall. In the DataFrame, 50% of the teams won, and 50% of the teams lost; the accuracy score is a sufficient measure as our model will not bias imbalances.

At the time of prediction, I only know the following information for each player: `firstdragon`, `firstherald`, `firstbaron`, `firsttower`, `goldat15`, and `killsat15`. These are all the statistics collected during the game, while the last two have been modified (as mentioned earlier) to meet imputation requirements. I will train our model based on the above features.

## Baseline Model
For the baseline model, I used a **Random Forest Classifier**, with the following three features: `firstdragon`, `firstherald`, and `killsat15`. These two features are both nominal-categorical, hence, I used the OneHotEncoder Transformer to encode their boolean values back into binary form for training, dropping the first columns to avoid multicollinearity. The `killsat15` column is transformed using the StandardScaler Transformer, as each match has a different length, which may cause the weights to drastically shift. I did not use any ordinal data.

After fitting the model, the accuracy score on the training data is **0.8664**. This means that our model is able to correctly predict **86.64%** of data. My model still has more room for improvement, such as engineering more features, adding model complexity, and tuning hyperparameters in the next section. 

## Final Model
In my final model, I added three more features: `firsttower`, `firstbaron`, and `goldat15`. For the first two features, I added these columns because I believe that in the LoL game, the milestone of destroying the first tower grants a team higher map control, and thus a winning advantage. Defeating the first baron grants a huge buff that allows the team that secured it to push waves extremely fast, and thus, destroying more towers. This objective typically requires an all-out team fight, that usually results in either one team eliminating most of the players on the other team, which provides high predictive power. As such, the last feature, gold at 15 minutes, represent a teams ability to retain an early quantitative advantage over the other team as a result of winning these teamfights and gaining gold through farming minions or objectives. This enables teams to develop their winning chances by buying more items for their champions, taking their opponents' times off the map, and ultimately convert these into more winning opportunities.

Furthermore, I believe these features are significantly more 'useful' than the baseline features, as in the early game, it possible for the first rift herald and first dragons to be 'traded' by each team, as they are on opposite sides of the map. Therefore, the odds of one team winning might not be clear to predict, if they had given up one objective for the other.

My final model also uses a **Random Forest Classifier** in alignment with the baseline model. The first two additional features added (`firsttower`, `firstbaron`) are categorical features, so I used the OneHotEncoder Transformer to transform them into binary features. The last feature `goldat15` is quantitative, so I used the StandardScaler Transformer to transform the columns into standard scale, because each match's duration varies widely, and therefore the statistics and overall feature importances could seem uninterpretable when unstandardized. 

For the hyperparameter-tuning pipeline, I used *GridSearchCV* to search over the set of all possible hyperparameters I had given the model. I used max depth, number of estimators, and criterion for the Random Forest Classifier. We are testing max depth of 2 through 100, with each of 5 steps. For the number of estimators, we are testing from 2 to 100, with each of 10 steps. I included gini and entropy as the two possible criterion for this model. After searching for the best hyperparameters, I found out that the best max depth is **7**, the best number of estimators is **82**, and the best criterion is **entropy**.

<iframe
  src="assets/fig14.html"
  width="800"
  height="400"
  frameborder="0"
></iframe>

The accuracy score is now **0.9367**, meaning our model is able to correctly predict **93.67%** of our data. This is a great improvement from the base model. We have achieved good improvement in our evaluation metric, and this improvement suggests that our adjustment to the model is effective in terms of prediction power.

## Fairness Analysis
In this section, we are going to assess whether our model is fair among different groups. The question we are trying to answer here is: **“Does my model perform worse for teams that have the first baron than it does for teams that do not?”** To answer this question, I performed a permutation test and examined the result of the difference in accuracy between the two groups.

Group `X` represents the teams that have first baron (`firstbaron == 1`), and Group `Y` represents those who do not have first baron (`firstbaron == 0`). My evaluation metric is accuracy, and the significance level is **0.05**.

The following are my hypotheses:

**Null hypothesis**: The model is fair. Its accuracy for teams who secured the first baron is the same as the accuracy for teams who do not.

**Alternative hypothesis**: The model is unfair. Its accuracy for teams who have secured the first baron is NOT the same as the accuracy for teams who have not secured the first baron.

**Test statistics**: The difference in accuracy between teams who secured first baron and teams that didn't.

After performing the permutation test, the resulting p-value was **0.8184**, which is larger than the 0.05 significance level. Consequently, we fail to reject the null hypothesis. This outcome implies that the model predicts game outcomes from both groups with statistically similar accuracy levels. Consequently, our model is fair, exhibiting no discernible bias towards the teams that secured first baron over the teams that didn't based on the specified criteria.