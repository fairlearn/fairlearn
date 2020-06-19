
Seed ideas for scenarios:
"""""""""""""""""""""""""
These scenarios are seed ideas for thinking through ways to approach fairness questions, and considering what may be similar or different in your own problem context.  They may be starting points for coming up with new ideas for example notebooks, or some may require other approaches that are out of scope for *fairlearn*.


Identifying potential tax fraud
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
You're a member of an analytics team in a European country, and brought in to consult about a project that has already started to scale the deployment of models for predicting which tax returns may require further investigation for fraud.  The team has used a model trained in other jurisdictions by a large predictive analytics supplier, and hopes that they can leverage this at a lower cost that would be required to invest in the capability in-house.
`Veale et al. (2018) <https://arxiv.org/pdf/1802.01029.pdf>`_

Debit card fraud investigation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
You're a data scientist at a Dutch financial services company, and your manager asks you to join an existing team.  This team has deploy a model trained on historical transaction data and now new debit transaction data is arriving.  For each new transaction, the model predicts whether it is potentially fraudulent and then will trigger an alert and inspection by human analysts.  The output that matters for the company is the final decision by the human analyst of whether to block the transaction, allow it but flag for further investigation by anothe team, or flag the transaction as normal.  False negatives mean clients can't get their money back (eg, in a phishing scheme), while false positives may overwhelm the team of human analysts or disrupt clients making legitimate purchases.
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

Compliance in customer service calls
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
You work on a team within financial services that is building a system to reduce the company's compliance risk from customer service phone calls.  Compliance risk includes when a company employee breaches confidentiality or engaging in instances of misrepresentation or fraud.  Another team has leveraged third-party services to transcribe call audio into text, and then extract features for each call related to the presence of specific keywords.  It's your team's role to take that vector of binary features, and build a system to estimate the compliance "risk score" for each call.  A team of internal analysts will use these risk scores to triage which calls to investigate further.
`vendor blog post (2020) <https://customers.microsoft.com/en-us/story/754840-kpmg-partner-professional-services-azure>`_

Facial verification of taxi drivers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Your team in a taxi company is collaborating on a new feature, "selfies for security," which asks drivers to periodically take pictures of themselves in between rides.  The intention is to reduce the company's risk in providing taxi's that are driven by someone who the company has not screened and approved.  These photos will be taken within taxi cars on cell phones, in a wide range of conditions with uncontrolled lighting throughout the day.  Another team in your company will generate the signal to "request a selfie" and your team is standing up a new service to process the photos through a third party facial verification vendor that returns a confidence score for how well the driver photo matches the last photo of the driver.  Your team's service then decides whether to allow the driver to start picking up riders, or to block the driver's account and flag it for investigation by a small team of analysts.
`taxi company blog post (2017) <https://eng.uber.com/real-time-id-check/>`_ and `vendor blog post (2019) <https://customers.microsoft.com/en-us/story/731196-uber>`_

Financial services product recommendations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
You work at a Canadian financial services company that makes financial product recommendations for consumers.  Other financial products describe their offerings and store them with your company.  Users come to the app, agree to share their credit history, and then after their identity is authenticated, your team builds a model to rank the financial products that are the best fits.
`financial services (2020) <https://customers.microsoft.com/en-us/story/734799-borrowell-financial-services-azure-machine-learning-devops-canada>`_

Customer Service triage, consulting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
You work at consulting company.  One of the services your company provides is setting up an single mailbox to receive incoming customer emails.  Your role is to collaborate with company to create a classification system for labeling emails in one of six categories.  The output of your system is then used to route the email to the correct department head.  To do this, you're using a third party keyword extraction system that the company has already set up, and can extract ~1000 binary features from an email.
`consulting blog post (2020) <https://customers.microsoft.com/en-us/story/774221-securex-professional-services-m365>`_

Job recommendations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
You work for a job recommendation product.  Background processes gather job posting, and submits them to a third party search indexing service.  When a user comes to the website and uploads their resume, the resume is processed and a set of job skills are extracted.  Your team works on the service that takes the set of job skills in a resume, and searches the job posting index service managed by a third party vendor.  Your team then provides the ranking of job postings that is ultimately shown to the user.
`company blog post <https://azure.microsoft.com/en-us/blog/using-azure-search-custom-skills-to-create-personalized-job-recommendations/>`_ and 

Alerting for first responder police officers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
You work at a company providing a service to police officers that accompanies queries typically run when a police officer is a first responder.  Three types of queries are run: driver’s license information, license plate information, and vehicle identification numbers.  When an officer presses a button on their radio and speaks a license plate number, within seconds they hear an alert tone that classifies whether the queries returned information that is low priority, sensitive but not urgent, or high priority (eg, a prior arrest record or a stolen vehicle).  The system relies on a third party language system to parse the audio and extract the license plate number, and then runs those queries through police department systems.  You work on the team building the classification system that chooses which of the three alert tones to play through the officers radio.
`company blog post <https://customers.microsoft.com/en-us/story/792324-motorola-solutions-manufacturing-azure-bot-service>`_

Choosing new retail sites
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
You work at a clothing company, as an analyst working to select the location for three new physical stores that will be opened in the next six months.  You're collaborating with a third-party vendor to estimate potential revenues at new site locations.  You've gathered data on past store openings, and shared it with the vendor, who has created a model that can estimate the potential revenue for the first two years of operation in new sites.  The vendor's model relies on data you've provided about your company's past openings, and other undisclosed data sources about retail sales, real-estate prices, foot traffic, etc.
`company blog post <https://customers.microsoft.com/en-us/story/816179-carhartt-retailers-azure>`_