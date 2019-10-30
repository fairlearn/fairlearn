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

export const ParityOptions: IParityOption[] = [
    {
        key: "m1",
        title: "Method 1",
        description: "uses science to get the answer",
        parityModes: [ParityModes.difference, ParityModes.ratio]
    },
    {
        key: "m2",
        title: "Method 2",
        description: "guesses at truth",
        parityModes: [ParityModes.difference, ParityModes.ratio]
    },
    {
        key: "m3",
        title: "Method 3",
        description: "guesses at truth",
        parityModes: [ParityModes.difference, ParityModes.ratio]
    },
    {
        key: "m4",
        title: "Method 4",
        description: "guesses at truth",
        parityModes: [ParityModes.difference, ParityModes.ratio]
    },
    {
        key: "m5",
        title: "Method 5",
        description: "guesses at truth",
        parityModes: [ParityModes.difference]
    },
    {
        key: "m6",
        title: "Method 6",
        description: "guesses at truth",
        parityModes: [ParityModes.difference]
    }
];
