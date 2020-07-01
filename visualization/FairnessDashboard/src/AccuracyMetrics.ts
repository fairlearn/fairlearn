import { localization } from './Localization/localization';

export interface IAccuracyOption {
    key: string;
    title: string;
    isMinimization: boolean;
    isPercentage: boolean;
    description?: string;
    tags?: string[];
    userVisible?: boolean;
    alwaysUpperCase?: boolean;
}

export const AccuracyOptions: { [key: string]: IAccuracyOption } = {
    accuracy_score: {
        key: 'accuracy_score',
        title: localization.Metrics.accuracyScore,
        description: localization.Metrics.accuracyDescription,
        isMinimization: false,
        isPercentage: true,
        userVisible: true,
    },
    precision_score: {
        key: 'precision_score',
        title: localization.Metrics.precisionScore,
        description: localization.Metrics.precisionDescription,
        isMinimization: false,
        isPercentage: true,
        userVisible: true,
    },
    recall_score: {
        key: 'recall_score',
        title: localization.Metrics.recallScore,
        description: localization.Metrics.recallDescription,
        isMinimization: false,
        isPercentage: true,
        userVisible: true,
    },
    zero_one_loss: {
        key: 'zero_one_loss',
        title: localization.Metrics.zeroOneLoss,
        description: localization.loremIpsum,
        isMinimization: true,
        isPercentage: true,
    },
    specificity_score: {
        key: 'specificity_score',
        title: localization.Metrics.specificityScore,
        description: localization.loremIpsum,
        isMinimization: false,
        isPercentage: true,
    },
    miss_rate: {
        key: 'miss_rate',
        title: localization.Metrics.missRate,
        description: localization.loremIpsum,
        isMinimization: true,
        isPercentage: true,
    },
    fallout_rate: {
        key: 'fallout_rate',
        title: localization.Metrics.falloutRate,
        description: localization.loremIpsum,
        isMinimization: true,
        isPercentage: true,
    },
    false_positive_over_total: {
        key: 'false_positive_over_total',
        title: localization.Metrics.falloutRate,
        description: localization.loremIpsum,
        isMinimization: true,
        isPercentage: true,
    },
    false_negative_over_total: {
        key: 'false_negative_over_total',
        title: localization.Metrics.falloutRate,
        description: localization.loremIpsum,
        isMinimization: true,
        isPercentage: true,
    },
    max_error: {
        key: 'max_error',
        title: localization.Metrics.maxError,
        description: localization.loremIpsum,
        isMinimization: true,
        isPercentage: false,
    },
    mean_absolute_error: {
        key: 'mean_absolute_error',
        title: localization.Metrics.meanAbsoluteError,
        description: localization.Metrics.meanAbsoluteErrorDescription,
        isMinimization: true,
        isPercentage: false,
        userVisible: true,
    },
    mean_squared_error: {
        key: 'mean_squared_error',
        title: localization.Metrics.meanSquaredError,
        description: localization.Metrics.mseDescription,
        isMinimization: true,
        isPercentage: false,
        userVisible: true,
    },
    mean_squared_log_error: {
        key: 'mean_squared_log_error',
        title: localization.Metrics.meanSquaredLogError,
        description: localization.loremIpsum,
        isMinimization: true,
        isPercentage: false,
    },
    median_absolute_error: {
        key: 'median_absolute_error',
        title: localization.Metrics.medianAbsoluteError,
        description: localization.loremIpsum,
        isMinimization: true,
        isPercentage: false,
    },
    balanced_root_mean_squared_error: {
        key: 'balanced_root_mean_squared_error',
        title: localization.Metrics.balancedRootMeanSquaredError,
        description: localization.Metrics.balancedRMSEDescription,
        isMinimization: true,
        isPercentage: false,
        userVisible: true,
    },
    average: {
        key: 'average',
        title: localization.Metrics.average,
        description: localization.loremIpsum,
        isMinimization: false,
        isPercentage: false,
    },
    selection_rate: {
        key: 'selection_rate',
        title: localization.Metrics.selectionRate,
        description: localization.loremIpsum,
        isMinimization: false,
        isPercentage: true,
    },
    overprediction: {
        key: 'overprediction',
        title: localization.Metrics.overprediction,
        description: localization.loremIpsum,
        isMinimization: true,
        isPercentage: false,
    },
    underprediction: {
        key: 'underprediction',
        title: localization.Metrics.underprediction,
        description: localization.loremIpsum,
        isMinimization: true,
        isPercentage: false,
    },
    r2_score: {
        key: 'r2_score',
        title: localization.Metrics.r2_score,
        description: localization.Metrics.r2Description,
        isMinimization: false,
        isPercentage: false,
        userVisible: true,
        alwaysUpperCase: true,
    },
    root_mean_squared_error: {
        key: 'root_mean_squared_error',
        title: localization.Metrics.rms_error,
        description: localization.Metrics.rmseDescription,
        isMinimization: true,
        isPercentage: false,
        userVisible: true,
        alwaysUpperCase: true,
    },
    auc: {
        key: 'auc',
        title: localization.Metrics.auc,
        description: localization.Metrics.aucDescription,
        isMinimization: false,
        isPercentage: false,
        userVisible: true,
    },
    balanced_accuracy_score: {
        key: 'balanced_accuracy_score',
        title: localization.Metrics.balancedAccuracy,
        description: localization.Metrics.balancedAccuracyDescription,
        isMinimization: false,
        isPercentage: true,
        userVisible: true,
    },
    f1_score: {
        key: 'f1_score',
        title: localization.Metrics.f1Score,
        description: localization.Metrics.f1ScoreDescription,
        isMinimization: false,
        isPercentage: false,
        userVisible: true,
        alwaysUpperCase: true,
    },
    log_loss: {
        key: 'log_loss',
        title: localization.Metrics.logLoss,
        description: localization.loremIpsum,
        isMinimization: true,
        isPercentage: false,
    },
};
