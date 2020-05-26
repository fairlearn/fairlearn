.. _faq:

Frequently Asked Questions
==========================

.. topic:: General Fairness Questions

    Where can I learn more about machine learning?
        Machine learning is an extremely broad topic, and the following list is but
        a small fraction of the possible sources of information:

        - A website
        - Another website
        - Maybe a book?

    Where can I learn more about fairness in machine learning?
        There are a variety of sources. For a high level overview, see books such as:

        - `Weapons of Math Destruction <https://weaponsofmathdestructionbook.com/>`_ by Cathy O'Neill
        - `Artificial Unintelligence <https://mitpress.mit.edu/books/artificial-unintelligence>`_ by Meredith Broussard
        - `Algorithms of Oppression <https://nyupress.org/9781479837243/algorithms-of-oppression/>`_ by Safiya Umoja Noble

        There are also some papers such as

        - `Big Data's Disparate Impact <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2477899##>`_ (Barocas & Selbst, 2016)
        - `Measures and Mismeasures of Fairness <https://5harad.com/papers/fair-ml.pdf>`_ (Corbett-Davies & Goel, 2018)
        - `Fairness and Abstraction in Sociotechnical Systems <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3265913>`_ (Selbst et al., 2018)
        - `A Survey on Bias and Fairness in Machine Learning <https://arxiv.org/abs/1908.09635>`_ (Mehrabi et al. 2019)

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
       Indeed, when used incorrectly, Fairlearn could produce model which is *less* fair.

       What Fairlearn can do is generate a model which has lower disparity as measured by
       a user-specified metric.
       Whether this makes the model fairer is outside the remit of Fairlearn - or indeed of
       machine learning in general.
       To determine whether the new model is fairer, the whole system of which it is a part
       needs to be considered in terms of the societal context in which it is used.

    What sort of fairness related harms can Fairlearn address?
        Fairlean is focussed on group fairness.
        That is, minimising the difference in some metric evalulated on
        specified subgroups in the data.
        This is certainly not the only way of looing at fairness, though.

    Can Fairlearn be used to detect bias in datasets?
        Not yet....

    Can Fairlearn recommend ways to make my model fairer?
        No. Fairlearn is not magic.

        Fairness is a social concept, and no technological solution can make
        things fair automatically.
        In order to use Fairlearn successfully, you first need to work out
        what fairness means for your particular problem.
        Once that is determined, you can then work out how to use the
        various algorithms and fairness metrics which Fairlearn supports
        to develop models which are *fairer by the standard you have chosen*.

    What unfairness mitigation techniques does Fairlearn support?
        ExpGrad, PostProcessing....

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
        information on how to add your contributions.


.. topic:: Fairlearn and Microsoft

    What is the relationship between Fairlearn and Microsoft?
        Fairlearn has grown from a project at Microsoft Research in New York City.

    How does Fairlearn relate to Microsoft's push for Responsible AI?
        Need text for this...

    What is the relationship between Fairlearn and AzureML?
        Those of us who work for Microsoft are enabling Fairlearn to integrate
        into the AzureML experience.
        However, this is only to provide access to the rich MLOps Lifecycle
        management capabilities of AzureML.
        No functionality in Fairlearn itself will be tied to AzureML.