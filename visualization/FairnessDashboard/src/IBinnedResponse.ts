import { RangeTypes } from 'mlchartlib';

export interface IBinnedResponse {
    hasError: boolean;
    array: Array<number | string>;
    labelArray: string[];
    featureIndex: number;
    rangeType: RangeTypes;
}
