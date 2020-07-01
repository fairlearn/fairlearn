import { localization } from './Localization/localization';

export interface IParityOption {
    key: string;
    title: string;
    description?: string;
    parityMetric: string;
    parityModes: ParityModes[];
}

export enum ParityModes {
    'difference',
    'ratio',
}

export const ParityOptions: { [key: string]: IParityOption } = {
    selection_rate: {
        key: 'selection_rate',
        title: localization.Metrics.parityDifference,
        description: localization.Metrics.parityDifferenceDescription,
        parityMetric: 'selection_rate',
        parityModes: [ParityModes.difference],
    },
    selection_rate_ratio: {
        key: 'selection_rate_ratio',
        title: localization.Metrics.parityRatio,
        description: localization.Metrics.parityRatioDescription,
        parityMetric: 'selection_rate',
        parityModes: [ParityModes.ratio],
    },
    zero_one_loss: {
        key: 'zero_one_loss',
        title: localization.Metrics.errorRateDifference,
        description: localization.Metrics.errorRateDifferenceDescription,
        parityMetric: 'zero_one_loss',
        parityModes: [ParityModes.difference],
    },
    recall_score: {
        key: 'recall_score',
        title: localization.Metrics.equalOpportunityDifference,
        description: localization.Metrics.equalOpportunityDifferenceDescription,
        parityMetric: 'recall_score',
        parityModes: [ParityModes.difference],
    },
};
