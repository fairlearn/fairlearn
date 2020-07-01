import { INumericRange, ICategoricalRange, RangeTypes } from 'mlchartlib';
import { IBinnedResponse } from './IBinnedResponse';
import _ from 'lodash';

export class BinnedResponseBuilder {
    public static buildCategorical(
        featureRange: INumericRange | ICategoricalRange,
        index: number,
        sensitiveFeatures: any[][],
    ): IBinnedResponse {
        if (featureRange.rangeType === RangeTypes.categorical) {
            return {
                hasError: false,
                array: (featureRange as ICategoricalRange).uniqueValues,
                featureIndex: index,
                rangeType: RangeTypes.categorical,
                labelArray: (featureRange as ICategoricalRange).uniqueValues,
            };
        }
        const uniqueValues = BinnedResponseBuilder.getIntegerUniqueValues(sensitiveFeatures, index);
        return {
            hasError: false,
            array: uniqueValues,
            featureIndex: index,
            rangeType: RangeTypes.categorical,
            labelArray: uniqueValues.map((num) => num.toString()),
        };
    }

    public static buildNumeric(
        featureRange: INumericRange,
        index: number,
        sensitiveFeatures: any[][],
        binCount?: number,
    ): IBinnedResponse {
        if (binCount === undefined) {
            if (featureRange.rangeType === RangeTypes.integer) {
                const uniqueValues = BinnedResponseBuilder.getIntegerUniqueValues(sensitiveFeatures, index);
                binCount = Math.min(5, uniqueValues.length);
            }
            if (binCount === undefined) {
                binCount = 5;
            }
        }
        const delta = featureRange.max - featureRange.min;
        if (delta === 0 || binCount === 0) {
            return {
                hasError: false,
                array: [featureRange.max],
                featureIndex: index,
                rangeType: RangeTypes.categorical,
                labelArray: [featureRange.max.toString()],
            };
        }
        // make uniform bins in these cases
        if (featureRange.rangeType === RangeTypes.numeric || delta < binCount - 1) {
            const binDelta = delta / binCount;
            const array = new Array(binCount).fill(0).map((unused, index) => {
                return index !== binCount - 1 ? featureRange.min + binDelta * (1 + index) : featureRange.max;
            });
            let prevMax = featureRange.min;
            const labelArray = array.map((num) => {
                const label = `${prevMax.toLocaleString(undefined, {
                    maximumSignificantDigits: 3,
                })} - ${num.toLocaleString(undefined, { maximumSignificantDigits: 3 })}`;
                prevMax = num;
                return label;
            });
            return {
                hasError: false,
                array,
                featureIndex: index,
                rangeType: RangeTypes.numeric,
                labelArray,
            };
        }
        // handle integer case, increment delta since we include the ends as discrete values
        const intDelta = delta / binCount;
        const array = new Array(binCount).fill(0).map((unused, index) => {
            if (index === binCount - 1) {
                return featureRange.max;
            }
            return Math.ceil(featureRange.min - 1 + intDelta * (index + 1));
        });
        let previousVal = featureRange.min;
        const labelArray = array.map((num) => {
            const label =
                previousVal === num
                    ? previousVal.toLocaleString(undefined, { maximumSignificantDigits: 3 })
                    : `${previousVal.toLocaleString(undefined, {
                          maximumSignificantDigits: 3,
                      })} - ${num.toLocaleString(undefined, { maximumSignificantDigits: 3 })}`;
            previousVal = num + 1;
            return label;
        });
        return {
            hasError: false,
            array,
            featureIndex: index,
            rangeType: RangeTypes.integer,
            labelArray,
        };
    }

    public static buildDefaultBin(
        featureRange: INumericRange | ICategoricalRange,
        index: number,
        sensitiveFeatures: any[][],
    ): IBinnedResponse {
        if (featureRange.rangeType === RangeTypes.categorical) {
            return BinnedResponseBuilder.buildCategorical(featureRange, index, sensitiveFeatures);
        }
        if (featureRange.rangeType === RangeTypes.integer) {
            const uniqueValues = BinnedResponseBuilder.getIntegerUniqueValues(sensitiveFeatures, index);
            if (uniqueValues.length <= BinnedResponseBuilder.UpperBoundUniqueIntegers) {
                return BinnedResponseBuilder.buildCategorical(featureRange, index, sensitiveFeatures);
            }
        }
        return BinnedResponseBuilder.buildNumeric(featureRange, index, sensitiveFeatures);
    }

    private static getIntegerUniqueValues(sensitiveFeatures: any[][], index: number): number[] {
        const column = sensitiveFeatures.map((row) => row[index]) as number[];
        return _.uniq(column).sort((a, b) => {
            return a - b;
        });
    }

    private static readonly UpperBoundUniqueIntegers = 10;
}
