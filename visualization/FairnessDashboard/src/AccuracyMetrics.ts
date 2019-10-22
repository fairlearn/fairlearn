import { localization } from "./Localization/localization";

export interface IAccuracyOption {
    key: string;
    title: string;
    description?: string;
    tags?: string[];
}

export const AccuracyOptions: {[key: string]: IAccuracyOption} = {
    "accuracy_score": {
        key: "accuracy_score",
        title: localization.Metrics.accuracyScore,
        description: localization.loremIpsum
    },
    "precision_score": {
        key: "precision_score",
        title: localization.Metrics.precisionScore,
        description: localization.loremIpsum
    },
    "recall_score": {
        key: "recall_score",
        title: localization.Metrics.recallScore,
        description: localization.loremIpsum
    },
    "zero_one_loss": {
        key: "zero_one_loss",
        title: localization.Metrics.zeroOneLoss,
        description: localization.loremIpsum
    },
    "specificity_score": {
        key: "specificity_score",
        title: localization.Metrics.specificityScore,
        description: localization.loremIpsum
    },
    "miss_rate": {
        key: "miss_rate",
        title: localization.Metrics.missRate,
        description: localization.loremIpsum
    },
    "fallout_rate": {
        key: "fallout_rate",
        title: localization.Metrics.falloutRate,
        description: localization.loremIpsum
    },
    "max_error": {
        key: "max_error",
        title: localization.Metrics.maxError,
        description: localization.loremIpsum
    },
    "mean_absolute_error": {
        key: "mean_absolute_error",
        title: localization.Metrics.meanAbsoluteError,
        description: localization.loremIpsum
    },
    "mean_squared_error": {
        key: "mean_squared_error",
        title: localization.Metrics.meanSquaredError,
        description: localization.loremIpsum
    },
    "mean_squared_log_error": {
        key: "mean_squared_log_error",
        title: localization.Metrics.meanSquaredLogError,
        description: localization.loremIpsum
    },
    "median_absolute_error": {
        key: "median_absolute_error",
        title: localization.Metrics.medianAbsoluteError,
        description: localization.loremIpsum
    }
};
