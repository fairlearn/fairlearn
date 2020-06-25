.. _faq:

Frequently asked questions
==========================

Where can I learn more about fairness in machine learning?
    For a high level view of the subject, please see the :ref:`further resources <further_resources>`
    section, where we provide links to a variety of books and papers.
    In our :ref:`user guide <user_guide>` we also have links to the papers describing
    Fairlearn's mitigation algorithms.

Why not just ignore the sensitive features?
    If the fairness criterion is for the result of the machine learning pipeline to be
    statistically independent of the sensitive features, simply ignoring the sensitive
    features will not usually work.
    Information is often redundantly encoded in datasets, and machine learning
    algorithms *will* uncover these links (it is what they are designed to do).
    For example, in the US, the ZIP code where a person lives is well correlated with their
    race.
    Even if the model is not provided with race as a feature, the model will pick up on it
    implicitly via the ZIP code (and other features).
    Worse, without having the race available in the dataset, it is not possible to determine
    whether any use of ZIP code was as a proxy for race, or for some other reason.

The model is unfair because the data are biased. Isn't it better to get better data?
    One first has to decide that is meant by 'biased data' and 'better data' in a particular
    context.
    Consider the example of a company seeking to build a tool for screening the resumes of
    job candidates.
    If the company has historically hired few women, then this underrepresentation will
    mean that any model will have less predictive power for women.
    On the other hand, if women have received systematically poorer reviews due to biased
    managers, then the model will appear to have good predictive power, but for incorrect
    labels.
    These are just two of the ways in which the data may be biased (they are not mutually
    exclusive either), and the processes for getting 'better data' will be different for
    each.
    Ultimately, obtaining 'better data' may not be practical.

Won't making a model fairer reduce its accuracy?
    There are often many machine learning models that achieve similar levels of accuracy
    or other performance metrics, but that dramatically differ in how they affect
    different subgroups.
    Mitigation algorithms seek to improve the fairness metrics without strongly affecting
    the accuracy, or more generally to navigate the trade-offs between performance and
    fairness metrics.

Can the mitigation algorithms in Fairlearn make my model fair?
    Naive use of the Fairlearn library is not going to result in fairer models.
    Indeed, when used incorrectly, these mitigation algorithms could produce a model which is *less* fair.
    What mitigation algorithms do is generate a model which has lower disparity as measured by
    a user-specified metric.
    Whether this makes the model fairer is outside the scope of the algorithm.
    To determine whether the new model is fairer, the whole system of which it is a part
    needs to be considered in terms of the societal context in which it is used.
    In Fairlearn, we aim to combine a Python library of disparity metrics and mitigation algorithms
    with a deep discussion of fairness in machine learning.
    By considering this extra information in the context of your particular machine learning system,
    you should be able to determine the most appropriate actions for your situation.

What sort of fairness related harms can the Fairlearn library address?
    Fairlearn concentrates on group fairness, meaning obtaining the minimum disparity on some
    metric evaluated on specified subgroups in the data.
    This is certainly not the only way of looking at fairness.
    For example, there is the concept of individual fairness (where decisions are evaluated
    on the level of individuals rather than groups), counter-factual fairness (e.g. does the
    decision change if an individual's gender is changed from male to female), and many
    others.
    We would welcome contributions which enabled Fairlearn to help address some of these other harms.
    Beyond these, there are also fairness concepts which are not amenable to
    mathematical (and hence algorithmic) expression - justice, due process and
    righting historic iniquities for example.

Can the Fairlearn library be used to detect bias in datasets?
    We do not have concrete plans for this at the present time.

Can the Fairlearn library recommend ways to make my model fairer?
    Fairness is a social concept, and no technological solution can make
    things fair automatically.
    In order to use the Fairlearn library successfully, you first need to work out
    what fairness means for your particular problem.
    We aim to provide a rich discussion of the topic of fairness in machine learning
    on this website, together with links to the literature.
    Once you have determined what fairness means for your AI system,
    you can then work out how to use the various algorithms and fairness metrics
    which Fairlearn supports to develop models which are
    *fairer by the standard you have chosen*.
    Even then, you should remember that some notions of fairness (such as justice
    or due process) are not amenable to mathematical expression or algorithmic
    mitigation.

What unfairness mitigation techniques does Fairlearn support?
    Please see our :ref:`mitigation` section.

Which ML libraries does Fairlearn support?
    We have generally followed conventions for `scikit-learn` in Fairlearn.
    However, support is not restricted to Estimators from `scikit-learn`.
    Any algorithm which provides (or can be wrapped to provide) `fit()` and
    `predict()` methods should work.

Does Fairlearn work for image and text data?
    We have not (yet) looked at using Fairlearn on image or text data.
    However, so long as the estimators used have `fit()` and `predict()` methods
    as required by Fairlearn, it should be possible to use Fairlearn on
    image or text models.

Is Fairlearn available in languages other than Python?
    For the moment, we only support Python >= 3.6

Can I contribute to Fairlearn?
    Absolutely! Please see our :ref:`Contributor Guide <contributor_guide>` for
    information on how to add your contributions. We welcome all contributions!

What is the relationship between Fairlearn and Microsoft?
    Fairlearn has grown from a project at Microsoft Research in New York City.
