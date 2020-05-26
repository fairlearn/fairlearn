.. _faq:

Frequently Asked Questions
==========================

.. topic:: General Fairness Questions

    1. Where can I learn more about fairness in machine learning

       There are a variety of sources. For a high level overview, see books such as:
            * Weapons of Math Destruction
            * Artificial Unintelligence

        There are also some papers such as
            * Paper 1
            * Paper 2

    1. Why not just ignore the sensitive features?

       Because ignoring the sensitive feature usually doesn't work.

       To elaborate, information is often redundantly encoded in datasets, and machine learning
       algorithms *will* ferret out these links (it is what they are designed to do).
       For example, in the US, the ZIP code where a person lives is well correlated with their
       race.
       Even if the model is not provided with race as a feature, the model will pick up on it
       implicitly via the ZIP code (and other features).
       Worse, without having the race available in the dataset, it is not possible to determine
       whether any use of ZIP code was as a proxy for race, or for some other reason.

    1. The model is unfair because the data are biased. Isn't it better to get better data?

       Yes, but gathering unbiased data might not always be practical.
       Fairlearn and similar packages offer techniques to redistribute harms between subgroups
       in the data.

.. topic:: Fairlearn Questions

    1. What sort of fairness related harms can Fairlearn address?

    1. Can Fairlearn recommend ways to make my model fairer?

       No. Fairlearn is not magic.

    1. Is Fairlearn available in languages other than Python?

    1. Can I contribute to Fairlearn?


.. topic:: Fairlearn and Microsoft

    1. What is the relationship between Fairlearn and Microsoft?

    1. What is the relationship between Fairlearn and AzureML?