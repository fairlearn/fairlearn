.. _faq:

Frequently asked questions
==========================

General fairness questions
--------------------------

Where can I learn more about fairness in machine learning?
    For a high level view of the subject, please see the :ref:`further reading <further_reading>`
    section, where we provide links to a variety of books and papers.
    In our :ref:`user guide <user_guide>` we also have links to the papers describing
    Fairlearn's mitigation algorithms.

Why not just ignore the sensitive features?
    Because ignoring the sensitive feature usually doesn't work.

    To elaborate, information is often redundantly encoded in datasets, and machine learning
    algorithms *will* uncover these links (it is what they are designed to do).
    For example, in the US, the ZIP code where a person lives is well correlated with their
    race.
    Even if the model is not provided with race as a feature, the model will pick up on it
    implicitly via the ZIP code (and other features).
    Worse, without having the race available in the dataset, it is not possible to determine
    whether any use of ZIP code was as a proxy for race, or for some other reason.

The model is unfair because the data are biased. Isn't it better to get better data?
    If a model is unfair due to biased data (note that this is not the only reason a model
    may be unfair), then obtaining unbiased data is definitely the best solution.
    However, doing so might not always be practical.
    In these cases, the mitigation techniques in Fairlearn and similar packages can redistribute
    harms between subgroups in the data.
    The extent to which this is 'fair' will depend on how the harms are redistributed, and the
    broader context of the system of which the model is a part.

Won't making a model fairer reduce its accuracy?
    Any technique which reduces disparity in some metric is most likely to impact other metrics.
    However, we have found empirically that it is usually possible to obtain substantial reductions
    in disparity (as measured by some suitable metric) at minimal cost in accuracy.
    Indeed, for certain subgroups in the data, accuracy may be improved dramatically.

Fairlearn questions
-------------------

Can Fairlearn make my model fair?
    Naive use of Fairlearn is not going to result in fairer models.
    Indeed, when used incorrectly, Fairlearn could produce a model which is *less* fair.

    What Fairlearn can do is generate a model which has lower disparity as measured by
    a user-specified metric.
    Whether this makes the model fairer is outside the remit of Fairlearn - or indeed of
    machine learning in general.
    To determine whether the new model is fairer, the whole system of which it is a part
    needs to be considered in terms of the societal context in which it is used.
    See, for example, "Fairness and Abstraction in Sociotechnical Systems" by Selbst et al.,
    listed on our :ref:`further_reading` page. 

What sort of fairness related harms can Fairlearn address?
    Fairlean concentrates on group fairness.
    That is, obtaining the minimum disparity on some metric evaluated on
    specified subgroups in the data.
    This is certainly not the only way of looking at fairness.
    For example, there is the concept of individual fairness (where decisions are evaluated
    on the level of individuals rather than groups) and counter-factual fairness (e.g. does the
    decision change if an individual's gender is changed from male to female).
    See :ref:`types_of_harms` for more information.
    Beyond these, there are also fairness concepts which are note amenable to
    mathematical (and hence algorithmic) expression - justice, due process and
    righting historic iniquities for example.

Can Fairlearn be used to detect bias in datasets?
    We do not have concrete plans for this at the present time.

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


Fairlearn and Microsoft
-----------------------

What is the relationship between Fairlearn and Microsoft?
    Fairlearn has grown from a project at Microsoft Research in New York City.
