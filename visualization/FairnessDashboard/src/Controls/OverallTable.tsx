import React from 'react';
import { List } from 'office-ui-fabric-react/lib/List';
import { Stack, StackItem } from 'office-ui-fabric-react/lib/Stack';
import { Text } from 'office-ui-fabric-react/lib/Text';
import { mergeStyleSets } from '@uifabric/styling';
import { Separator } from 'office-ui-fabric-react/lib/Separator';
import { localization } from '../Localization/localization';
import { OverallTableStyles } from './OverallTable.styles';

export interface IOverallTableProps {
    binValues: number[];
    formattedBinValues: string[][];
    binLabels: string[];
    metricLabels: string[];
    expandAttributes: boolean;
    overallMetrics: string[];
    binGroup: string;
}

interface IBinItem {
    title: string;
    score: string;
    isMin: boolean;
    isMax: boolean;
}

export class OverallTable extends React.PureComponent<IOverallTableProps> {
    private static readonly classNames = mergeStyleSets({});

    public render(): React.ReactNode {
        const styles = OverallTableStyles();
        let minIndexes = [];
        let maxIndexes = [];
        let minValue = Number.MAX_SAFE_INTEGER;
        let maxValue = Number.MIN_SAFE_INTEGER;
        this.props.binValues.forEach((value, index) => {
            if (value >= maxValue) {
                if (value === maxValue) {
                    maxIndexes.push(index);
                } else {
                    maxIndexes = [index];
                    maxValue = value;
                }
            }
            if (value <= minValue) {
                if (value === minValue) {
                    minIndexes.push(index);
                } else {
                    minIndexes = [index];
                    minValue = value;
                }
            }
        });
        return (
            <div className={styles.frame}>
                <div className={styles.groupCol}>
                    <div className={styles.groupLabel}>{/*this.props.binGroup*/}</div>
                    <div className={styles.flexCol}>
                        <div className={styles.binBox}>
                            <div className={styles.binTitle}>{localization.Report.overallLabel}</div>
                        </div>
                        {this.props.binLabels.map((label, index) => {
                            if (this.props.expandAttributes)
                                return (
                                    <div className={styles.binBox} key={index}>
                                        <div className={styles.binLabel}>{label}</div>
                                        {/* <Stack horizontal>
                                    {minIndexes.includes(index) && <div className={styles.minMaxLabel}>{localization.Report.minTag}</div>}
                                    {maxIndexes.includes(index) && <div className={styles.minMaxLabel}>{localization.Report.maxTag}</div>}
                                </Stack> */}
                                    </div>
                                );
                        })}
                    </div>
                </div>
                {this.props.metricLabels.map((metric, index) => {
                    return (
                        <div className={styles.metricCol}>
                            <div className={styles.metricLabel}>{metric}</div>
                            <div className={styles.flexCol}>
                                <div className={styles.metricBox}>{this.props.overallMetrics[index]}</div>
                                {this.props.formattedBinValues[index].map((value, index) => {
                                    if (this.props.expandAttributes)
                                        return (
                                            <div className={styles.metricBox} key={index}>
                                                {value !== undefined ? value : 'empty'}
                                            </div>
                                        );
                                })}
                            </div>
                        </div>
                    );
                })}
            </div>
        );
    }
}
