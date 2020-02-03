export interface IParityOption {
    key: string;
    title: string;
    description?: string;
    parityModes: ParityModes[];
}

export enum ParityModes {
    "difference",
    "ratio"
}

export const ParityOptions: {[key: string]: IParityOption} = {
    "parity_difference": {
        key: "parity_difference",
        title: "Parity difference",
        description: "Parity differenece",
        parityModes: [ParityModes.difference, ParityModes.ratio]
    },
    "parity_ratio": {
        key: "parity_ratio",
        title: "Parity ratio",
        description: "Parity ratio",
        parityModes: [ParityModes.difference, ParityModes.ratio]
    },
    "error_rate_difference": {
        key: "error_rate_difference",
        title: "Error rate difference",
        description: "Error rate difference",
        parityModes: [ParityModes.difference, ParityModes.ratio]
    },
    "equal_opportunity_difference": {
        key: "equal_opportunity_difference",
        title: "Equal opportunity difference",
        description: "Equal opportunity difference",
        parityModes: [ParityModes.difference, ParityModes.ratio]
    }
};
