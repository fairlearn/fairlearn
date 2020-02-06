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
    "selection_rate": {
        key: "selection_rate",
        title: "Parity difference",
        description: "Parity differenece",
        parityModes: [ParityModes.difference]
    },
    // "selction_rate": {
    //     key: "selction_rate",
    //     title: "Parity ratio",
    //     description: "Parity ratio",
    //     parityModes: [ParityModes.ratio]
    // },
    "zero_one_loss": {
        key: "zero_one_loss",
        title: "Error rate difference",
        description: "Error rate difference",
        parityModes: [ParityModes.difference]
    },
    "recall_score": {
        key: "recall_score",
        title: "Equal opportunity difference",
        description: "Equal opportunity difference",
        parityModes: [ParityModes.difference]
    }
};
