.. _faq:

Frequently asked questions
==========================

.. currentmodule:: fairlearn

Where can I learn more about fairness in machine learning?
    Please review :ref:`further resources <further_resources>`,
    where we provide links to various materials that we have found helpful.
    Also, in our :ref:`user guide <user_guide>`, we link to the papers describing
    the algorithms implemented in Fairlearn.

Why not just ignore the sensitive features?
    If your goal is to train a model whose predictions are statistically
    independent of the sensitive features, then it is not enough to simply
    ignore the sensitive features.
    Information is often redundantly encoded across several features, and
    machine learning algorithms *will* uncover these links (it is what they
    are designed to do).
    For example, in the US, the ZIP code where a person lives is well
    correlated with their race.
    Even if the model is not provided with race as a feature, the model will
    pick up on it implicitly via the ZIP code (and other features).
    Worse, without having the race available in the dataset, it is hard to
    assess the  model's impact across different groups defined by race or by
    race intersected with other demographic features.

The model is unfair because the data are biased. Isn't it better to get better data?
    The answer to this question depends on what is meant by 'unfair',
    'biased data', and 'better data' in any particular context.
    Consider the example of a company seeking to build a tool for screening
    the resumes of job candidates.
    The company is planning to use their internal job evaluation data and
    train a model to predict job evaluations of the applicants; those with
    higher predictions will be ranked higher by the screening tool.
    This setup might present several fairness issues:

    - If the company has historically hired few women, there will be fewer of
      them in the training data set, and so a trained model may be less
      accurate for them.
    - The choice of features also affects the accuracy of the model.
      The features that are predictive for one group of applicants might not
      be as predictive for another group, and so more data will not
      necessarily improve the accuracy.
    - The accuracy of a model might not mean that the model is fair.
      If women have received systematically poorer reviews due to biased
      managers or worse workplace conditions, then the model might appear to
      be accurate, but the choice of the label (in this case, job evaluation)
      does not accurately reflect the applicants' potential.

    These are just three ways how the data may be 'biased', and they are not
    mutually exclusive. The processes for getting 'better data' will be
    different for each. In some of these cases, obtaining 'better data' may
    not be practical, but it might still be possible to use some mitigation
    algorithms.

Why am I seeing fairness issues, even though my data are reflective of the general population?
    Machine learning models often perform poorly for subgroups which are
    poorly  represented.
    What constitutes poor representation is context specific, and may well be
    affected by historical misrepresentation (consider the example above, of a
    company which had previously hired few women).
    For this reason, balanced sampling is generally better for ML than
    population sampling.
    On a related point, this is why it is important to consider multiple
    fairness metrics, and how they vary across different subgroups.

Won't making a model fairer reduce its accuracy?
    There are often many machine learning models that achieve similar levels
    of accuracy or other performance metrics, but that dramatically differ in
    how they affect different subgroups.
    Mitigation algorithms seek to improve the fairness metrics without
    strongly affecting the accuracy, or more generally to navigate the
    trade-offs between performance and fairness metrics.

Can the mitigation algorithms in Fairlearn make my model fair?
    There are many ways in which a model can be unfair. Fairlearn mitigation
    algorithms only address some of them: those that can be quantified by our
    supported fairness metrics.
    However, to assess whether the new model is fairer, it is important to
    consider not only the fairness metrics, but also the societal and
    technical context in which the model is applied.

I've got improved data, trained and mitigated new models, checked all the metrics... am I done?
    Firstly, always remember that there is more to fairness than technical
    details such as metrics - fairness is a *sociotechnical* problem.
    Even if these are all considered at training time, the ML lifecycle
    doesn't end when a model is deployed.
    Models need to be monitored in production.
    On a technical level, this means checking for data drift within the
    vulnerable subgroups identified during the fairness analysis.
    However the societal aspects need to be considered as well, for example:

    - Are the actual harms as expected, both in the nature of the harms and
      their distribution?
    - Have the users of the model (who may not be the subjects of the model)
      adjusted their usage patterns? This is sometimes called 'strategic
      behavior.'

What sort of fairness-related harms can the Fairlearn library address?
    We currently focus on two kinds of harms:

    - *Allocation harms.*
      These harms can occur when AI systems extend or withhold opportunities,
      resources, or information. Some of the key applications are in hiring,
      school admissions, and lending.
    - *Quality-of-service harms.* Quality of service refers to whether a
      system works as well for one person as it does for another, even if no
      opportunities, resources, or information are extended or withheld.

Can the Fairlearn library be used to detect bias in datasets?
    We do not have concrete plans for this at the present time.

Can the Fairlearn library recommend ways to make my model fairer?
    Right now we do not have an automated tool that would help you decide
    which mitigation algorithm to use. Our focus is on expanding the
    documentation and examples to highlight when each of the algorithms might
    be more applicable.
    Note that model training is just one step in the AI development and
    deployment lifecycle, and other steps, such as data gathering and
    curation, or monitoring and debugging of the deployed system, may be
    better places of intervention to improve the fairness of an AI system.

What unfairness mitigation techniques does Fairlearn support?
    Please see our :ref:`mitigation` section.

Which ML libraries does Fairlearn support?
    We have generally followed conventions of `scikit-learn`.
    However, our mitigation algorithms can be used to augment
    any ML algorithms that provide (or can be wrapped to provide) `fit()` and
    `predict()` methods. Also, any classification or regression
    algorithm can be evaluated using our metrics.

Does Fairlearn support multiclass classification?
    On the assessment side, Fairlearn's :class:`metrics.MetricFrame`
    can be used with any metrics for supervised learning, including those for
    multiclass classification.
    For example, it is possible to pass
    :py:func:`sklearn.metrics.accuracy_score` or
    :py:func:`sklearn.metrics.confusion_matrix` as the metric functions, and
    supply multiclass data for :code:`y_true` and :code:`y_pred`.
    We give an example with multiclass data in the :ref:`user guide <assessment>`.
    There are
    `ongoing discussions within the community <https://github.com/fairlearn/fairlearn/issues/752>`_
    to add more extensive support to Fairlearn's assessment capabilities.
    If you have thoughts feel free to add them to the discussion.

    On the mitigation side, our algorithms
    :class:`reductions.ExponentiatedGradient` and
    :class:`reductions.GridSearch`
    support :ref:`bounded group loss <bounded_group_loss>` constraints, which
    are applicable to any supervised learning setting, including multiclass
    classification.

Does Fairlearn support multiple and non-binary sensitive features?
    Fairlearn's assessment capabilities support sensitive features with more
    than two values as well as multiple sensitive features.
    Our :ref:`user guide <assessment>` has examples for both of
    these cases.
    The mitigation techniques all support mitigation with non-binary and
    multiple sensitive features as well. For a full list of techniques
    please refer to the :ref:`user guide section on mitigation <mitigation>`.

Does Fairlearn work for image and text data?
    We have not (yet) looked at using Fairlearn on image or text data.
    However, so long as the image or text classifier provide
    :code:`fit()` and :code:`predict()` methods
    as required by Fairlearn, it should be possible to use them
    with Fairlearn mitigation algorithms. Also, any classification or
    regression algorithm can be evaluated using our metrics (regardless of the
    data it is operating on).

Is Fairlearn available in languages other than Python?
    For the moment, we only support Python >= 3.8

Can I contribute to Fairlearn?
    Absolutely! Please see our :ref:`contributor guide <contributor_guide>` to
    see how. We welcome all contributions!

What is the relationship between Fairlearn and Microsoft?
    Fairlearn has grown from a project at Microsoft Research in New York City.

What happened to the :code:`FairlearnDashboard`?
    The Fairlearn dashboard was a Jupyter notebook widget for assessing how a
    model's predictions impact different groups as defined by sensitive features, and
    also for comparing multiple models in terms of various fairness and performance
    metrics.

    The :code:`FairlearnDashboard` is no longer being developed as
    part of Fairlearn. Instead, it has found a new home at Microsoft with the name
    :code:`FairnessDashboard`. For more information on how to use it refer to
    `https://github.com/microsoft/responsible-ai-toolbox <https://github.com/microsoft/responsible-ai-toolbox>`_.
    Fairlearn provides some of the existing functionality through
    :code:`matplotlib`-based visualizations. Refer to the :ref:`plot_metricframe` section.
