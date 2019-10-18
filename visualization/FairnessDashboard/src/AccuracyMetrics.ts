export interface IAccuracyOption {
    key: string;
    title: string;
    description?: string;
    tags?: string[];
}

export const AccuracyOptions: {[key: string]: IAccuracyOption} = {
    "accuracy_score": {
        key: "accuracy_score",
        title: "Method 1",
        description: "uses science to get the answer"
    },
    "precision_score": {
        key: "precision_score",
        title: "Method 2",
        description: "guesses at truth"
    },
    "recall_score": {
        key: "recall_score",
        title: "Method 3",
        description: "guesses at truth"
    },
    "zero_one_loss": {
        key: "zero_one_loss",
        title: "Method 4",
        description: "guesses at truth"
    },
    "max_error": {
        key: "max_error",
        title: "Method 5",
        description: "guesses at truth"
    },
    "mean_absolute_error": {
        key: "mean_absolute_error",
        title: "Method 6",
        description: "guesses at truth"
    }
};
