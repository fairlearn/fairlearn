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
    "difference": {
        key: "difference",
        title: "Accuracy",
        description: "Disparity on Accuracy",
        parityModes: [ParityModes.difference]
    },
    "ratio" : {
        key: "ratio",
        title: "Prediction",
        description: "Disparity in Prediction",
        parityModes: [ParityModes.ratio]
    }
};
