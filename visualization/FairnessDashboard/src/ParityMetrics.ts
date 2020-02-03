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
        title: "Difference",
        description: "Disparity using difference",
        parityModes: [ParityModes.difference]
    },
    "ratio" : {
        key: "ratio",
        title: "Ratio",
        description: "Disparity using ratio",
        parityModes: [ParityModes.ratio]
    }
};
