.. _faq:

Frequently Asked Questions
==========================

.. topic:: General Fairness Questions

    Where can I learn more about machine learning?

        List some resources

    Where can I learn more about fairness in machine learning?

        There are a variety of sources. For a high level overview, see books such as:
            - Weapons of Math Destruction
            - Artificial Unintelligence

        There are also some papers such as
            - Paper 1
            - Paper 2

    Why not just ignore the sensitive features?

        Because ignoring the sensitive feature usually doesn't work.

        To elaborate, information is often redundantly encoded in datasets, and machine learning
        algorithms *will* ferret out these links (it is what they are designed to do).
        For example, in the US, the ZIP code where a person lives is well correlated with their
        race.
        Even if the model is not provided with race as a feature, the model will pick up on it
        implicitly via the ZIP code (and other features).
        Worse, without having the race available in the dataset, it is not possible to determine
        whether any use of ZIP code was as a proxy for race, or for some other reason.

    The model is unfair because the data are biased. Isn't it better to get better data?

        Yes, but gathering unbiased data might not always be practical.
        Fairlearn and similar packages offer techniques to redistribute harms between subgroups
        in the data.

    Won't making a model fairer reduce its accuracy?

        Any technique which reduces disparity in some metric is most likely to impact other metrics.
        However, we have found empirically that it is usually possible to obtain substantial reductions
        in disparity (as measured by some suitable metric) at minimal cost in accuracy.
        Indeed, for certain subgroups in the data, accuracy may be improved dramatically.

.. topic:: Fairlearn Questions

    Can Fairlearn make my model fair?

       Naive use of Fairlearn is not going to result in fairer models.
       Indeed, when used incorrectly, Fairlearn could produce model which is _less_ fair.

       What Fairlearn can do is generate a model which has lower disparity as measured by
       a user-specified metric.
       Whether this makes the model fairer is outside the remit of Fairlearn - or indeed of
       machine learning in general.
       To determine whether the new model is fairer, the whole system of which it is a part
       needs to be considered in terms of the societal context in which it is used.

    What sort of fairness related harms can Fairlearn address?

    Can Fairlearn be used to detect bias in datasets?

    Can Fairlearn recommend ways to make my model fairer?

        No. Fairlearn is not magic.

    What unfairness mitigation techniques does Fairlearn support?

    Which ML libraries does Fairlearn support?

        We have generally followed conventions for `scikit-learn` in Fairlearn.
        However, support is not restricted to Estimators from `scikit-learn`.
        Any algorithm which provides (or can be wrapped to provide) `fit()` and
        `predict()` methods should work.

    Is Fairlearn available in languages other than Python?

        For the moment, we only support Python >= 3.6

    Can I contribute to Fairlearn?

        Absolutely! Please see our Contributing Guide.


.. topic:: Fairlearn and Microsoft

    What is the relationship between Fairlearn and Microsoft?

        Fairlearn has grown from a project at Microsoft Research in New York City.

    How does Fairlearn relate to Microsoft's push for Responsible AI?

    What is the relationship between Fairlearn and AzureML?

        Those of us who work for Microsoft are enabling Fairlearn to integrate
        into the AzureML experience.
        However, this is only to provide access to the rich MLOps Lifecycle
        management capabilities of AzureML.
        No functionality in Fairlearn itself will be tied to AzureML.