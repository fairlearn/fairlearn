export interface IAccuracyOption {
    key: string;
    title: string;
    description?: string;
    tags?: string[];
}

export const AccuracyOptions: IAccuracyOption[] = [
    {
        key: "m1",
        title: "Method 1",
        description: "uses science to get the answer"
    },
    {
        key: "m2",
        title: "Method 2",
        description: "guesses at truth"
    },
    {
        key: "m3",
        title: "Method 3",
        description: "guesses at truth"
    },
    {
        key: "m4",
        title: "Method 4",
        description: "guesses at truth"
    },
    {
        key: "m5",
        title: "Method 5",
        description: "guesses at truth"
    },
    {
        key: "m6",
        title: "Method 6",
        description: "guesses at truth"
    }
];
