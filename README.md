# Data_Science_Space
Practice various data science problems

### Credit:
I took various dataset from https://data-flair.training/blogs/

I don't strictly follow the instructions on the guidance nor the source code. I follow my own flow design and intuitive, and make my own experiments from there.

All the datasets seems to be "clean", that says, without any data preparation step, we are good to go. In my point of view, it is a let-down if we are going to work with real-life data science projects, which involve heavily on this step. I will try to find the raw data somewhere else... 

I managed to create codes that are highly re-useable without any heavy modifications from project to project. Their architecture follows the same philosophy, so although there are modifications / additional modules depending on the project or the dataset, the main structure of the code doesn't need to change.

Note that on each type of data, depending on their ultimate goals and applications on specific business they serve, one might want to process data feature engineering somehow, either breakdown a feature to smaller features, or combine multiple features into more useful single feature. That's various from project to project, what already achieved in one project does not guarantee its success for the others. So keep it in mind

Therefore, the data preprocessing might vary heavily, that's due to the feature engineering process. Other steps, like filling missing values, dropping unnecessary columns, ... Might rely on human decision, not automatic by machine (although some auto ML / data science tools might solve those already, but it would be best if human can take the final decision)