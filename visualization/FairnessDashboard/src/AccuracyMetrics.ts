import { localization } from "./Localization/localization";

export interface IAccuracyOption {
    key: string;
    title: string;
    isMinimization: boolean;
    isPercentage: boolean;
    description?: string;
    tags?: string[];
}

export const AccuracyOptions: {[key: string]: IAccuracyOption} = {
    "accuracy_score": {
        key: "accuracy_score",
        title: localization.Metrics.accuracyScore,
        description: localization.Metrics.accuracyDescription,
        isMinimization: false,
        isPercentage: true
    },
    "precision_score": {
        key: "precision_score",
        title: localization.Metrics.precisionScore,
        description: localization.Metrics.precisionDescription,
        isMinimization: false,
        isPercentage: true
    },
    "recall_score": {
        key: "recall_score",
        title: localization.Metrics.recallScore,
        description: localization.Metrics.recallDescription,
        isMinimization: false,
        isPercentage: true
    },
    "zero_one_loss": {
        key: "zero_one_loss",
        title: localization.Metrics.zeroOneLoss,
        description: localization.loremIpsum,
        isMinimization: true,
        isPercentage: true
    },
    "specificity_score": {
        key: "specificity_score",
        title: localization.Metrics.specificityScore,
        description: localization.loremIpsum,
        isMinimization: false,
        isPercentage: true
    },
    "miss_rate": {
        key: "miss_rate",
        title: localization.Metrics.missRate,
        description: localization.loremIpsum,
        isMinimization: true,
        isPercentage: true
    },
    "fallout_rate": {
        key: "fallout_rate",
        title: localization.Metrics.falloutRate,
        description: localization.loremIpsum,
        isMinimization: true,
        isPercentage: true
    },
    "false_positive_over_total": {
        key: "false_positive_over_total",
        title: localization.Metrics.falloutRate,
        description: localization.loremIpsum,
        isMinimization: true,
        isPercentage: true
    },
    "false_negative_over_total": {
        key: "false_negative_over_total",
        title: localization.Metrics.falloutRate,
        description: localization.loremIpsum,
        isMinimization: true,
        isPercentage: true
    },
    "max_error": {
        key: "max_error",
        title: localization.Metrics.maxError,
        description: localization.loremIpsum,
        isMinimization: true,
        isPercentage: false
    },
    "mean_absolute_error": {
        key: "mean_absolute_error",
        title: localization.Metrics.meanAbsoluteError,
        description: localization.loremIpsum,
        isMinimization: true,
        isPercentage: false
    },
    "mean_squared_error": {
        key: "mean_squared_error",
        title: localization.Metrics.meanSquaredError,
        description: localization.loremIpsum,
        isMinimization: true,
        isPercentage: false
    },
    "mean_squared_log_error": {
        key: "mean_squared_log_error",
        title: localization.Metrics.meanSquaredLogError,
        description: localization.loremIpsum,
        isMinimization: true,
        isPercentage: false
    },
    "median_absolute_error": {
        key: "median_absolute_error",
        title: localization.Metrics.medianAbsoluteError,
        description: localization.loremIpsum,
        isMinimization: true,
        isPercentage: false
    },
    "balanced_root_mean_squared_error": {
        key: "balanced_root_mean_squared_error",
        title: localization.Metrics.balancedRootMeanSquaredError,
        description: localization.loremIpsum,
        isMinimization: true,
        isPercentage: false
    },
    "average": {
        key: "average",
        title: localization.Metrics.average,
        description: localization.loremIpsum,
        isMinimization: false,
        isPercentage: false
    },
    "selection_rate": {
        key: "selection_rate",
        title: localization.Metrics.selectionRate,
        description: localization.loremIpsum,
        isMinimization: false,
        isPercentage: true
    },
    "overprediction": {
        key: "overprediction",
        title: localization.Metrics.overprediction,
        description: localization.loremIpsum,
        isMinimization: true,
        isPercentage: false
    },
    "underprediction": {
        key: "underprediction",
        title: localization.Metrics.underprediction,
        description: localization.loremIpsum,
        isMinimization: true,
        isPercentage: false
    }
};
