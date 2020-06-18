
Seed ideas for scenarios:
"""""""""""""""""""""""""
These scenarios are seed ideas for thinking through ways to approach fairness questions, and considering what may be similar or different in your own problem context.  They may be starting points for coming up with new ideas for example notebooks, or some may require other approaches that are out of scope for *fairlearn*.


Identifying potential tax fraud
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
You're a member of an analytics team in a European country, and brought in to consult about a project that has already started to scale the deployment of models for predicting which tax returns may require further investigation for fraud.  The team has used a model trained in other jurisdictions by a large predictive analytics supplier, and hopes that they can leverage this at a lower cost that would be required to invest in the capability in-house.
`Veale et al. (2018) <https://arxiv.org/pdf/1802.01029.pdf>`_

Credit card fraud investigation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
You're a data scientist at a financial services company, and your manager asks you to join an existing team.  This team has deploy a model trained on historical transaction data and now new data is arriving.  For each new transaction, the model predicts whether it is potentially fraudulent and then will trigger an alert and inspection by human analysts.  The output that matters for the company is the final decision by the human analyst of whether to block the transaction, allow it but flag for further investigation by anothe team, or flag the transaction as normal.
`Weerts et al. (2019) <https://arxiv.org/abs/1907.03334>`_

Measuring brand sentiment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
You're a member of a team trying to measure brand sentiment from online comments and reviews.  The team hopes to use an existing language model, a third-party service for flagging abusive comments, and then train a more targeted sentiment classifier for your brand on top.
`Hutchinson et al. (2020) <https://arxiv.org/pdf/2005.00813.pdf>`_

Employment evaluations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
A potential client asks you if ML can help with predicting job candidates' suitability for jobs based on a combination of personality tests and body language
`Rhagavan et al. (2019) <https://arxiv.org/pdf/1906.09208.pdf>`_

Rankings for image search
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
You're on a team working on improving an image search system after receiving some complaints from users related to fairness.  Users often use this system to find a selection of stock images to use when making multimedia presentations.  In this system, requests start with some information about the context and user creating the query, and your team is trying to incorporate ideas about fairness like diversity and inclusision into how search results are ranked.
`Mitchell et al. (2020) <https://arxiv.org/pdf/2002.03256.pdf>`_

Sales leads for car loans
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
You work at CarCorp, a company that collects special financing data: information on people who need car financing but have either low credit scores or limited credit histories, and sells this data to auto dealer as sales leads.  CarCorp  serves dealers across the United States.  A new project manager asks about leveraging data science to “improve the quality” of leads so that dealers to not churn.  CarCorp has a large amount of historical lead data (2 million leads in 2017 alone), but relatively less data on which leads had been approved for special financing (let alone why the loan was approved).
`Passi and Barocas (2019) shttps://arxiv.org/ftp/arxiv/papers/1901/1901.02547.pdf>`_

Predictive policing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
You are a contractor working with the police department in a large city.  One of the project leaders in the department would like to construct a risk score for people who are known gang members engaging in knife crime.  It's important to them that they can understand what the model is doing, and are way that any model will pick up on protected characteristics.
`Veale et al. (2018) <https://arxiv.org/pdf/1802.01029.pdf>`_

Scheduling maintenance within a factory
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
You work within a manufacturing company, and are starting a new project that will create a schedule assigning employees to check and update certain components of the machinery to prevent critical operation failures. The component assignment is based on data that show how often different components have worn out and broken down in the past.
`Kyung Lee (2018) <AlgoManagePerception.pdf>`_

Child protective services hotline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
You're collaborating with the child protective services agency as part of a county government in the US.  The agency is redesigning the intake flow for reports of potential child abuse or neglect, and wants to discuss if a predictive analytics system could help them improve this system.
`Brown et al. (2019) <https://www.andrew.cmu.edu/user/achoulde/files/accountability_final_balanced.pdf>`_ and `Chouldechova et al. (2018) <http://proceedings.mlr.press/v81/chouldechova18a/chouldechova18a.pdf>`_