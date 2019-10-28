import { AccuracyOptions } from "./AccuracyMetrics";

export class FormatMetrics {
    public static formatNumbers = (value: number, key: string, isRatio: boolean = false): string => {
    if (value === null || value === undefined) {
        return NaN.toString();
    }
    const styleObject = {maximumSignificantDigits: 3};
    if (AccuracyOptions[key].isPercentage && !isRatio) {
        (styleObject as any).style = "percent";
    }
    return value.toLocaleString(undefined, styleObject);
}
}