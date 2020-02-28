import React from "react";
import { List } from "office-ui-fabric-react/lib/List";
import { Stack, StackItem } from "office-ui-fabric-react/lib/Stack";
import { Text } from "office-ui-fabric-react/lib/Text";
import { mergeStyleSets } from "@uifabric/styling";
import { Separator } from "office-ui-fabric-react/lib/Separator";
import { localization } from "../Localization/localization";

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
    private static readonly classNames = mergeStyleSets({
        minMaxLabel: {
            padding: "1px 9px",
            marginTop: "4px",
            color: "#FFFFFF",
            fontSize: "10px",
            lineHeight: "20px",
            fontWeight: "400",
            backgroundColor: "#999999"
        },
        groupCol: {
            display: "inline-flex",
            flexDirection:"column",
            height: "100%",
            width: "max-content"
        },
        groupLabel: {
            color: "#333333",
            fontSize: "12px",
            lineHeight: "12px",
            fontWeight: "500",
            height: "26px"
        },
        flexCol: {
            display: "flex",
            flex: 1,
            flexDirection: "column",
            borderTop: "0.5px solid #CCCCCC",
            borderBottom: "0.5px solid #CCCCCC"
        },
        binBox: {
            flex: 1,
            // minWidth: "100px",
            // maxWidth: "200px",
            // width: "max-content",
            width: "130px",
            display: "flex",
            flexDirection: "column",
            justifyContent: "center",
            borderBottom: "0.5px dashed #CCCCCC"
        },
        binTitle: {
            color: "#333333",
            fontSize: "15px",
            lineHeight: "18px",
            fontWeight: "400"
        },
        sensitiveAttributes: {
            color: "#333333",
            fontSize: "12px",
            lineHeight: "12px",
            fontWeight: "500",
            height: "26px",
            paddingLeft: "10px"
        },
        metricCol: {
            display: "inline-flex",
            flexDirection:"column",
            height: "100%",
            width: "120px"
        },
        metricLabel: {
            color: "#333333",
            fontSize: "12px",
            lineHeight: "12px",
            fontWeight: "500",
            height: "26px",
            paddingLeft: "10px"
        },
        metricBox: {
            flex: 1,
            paddingLeft: "10px",
            color: "#333333",
            fontSize: "32px",
            lineHeight: "39px",
            fontWeight: "100",
            display: "flex",
            flexDirection: "column",
            justifyContent: "center",
            // borderLeft: "0.5px dashed #CCCCCC",
            borderBottom: "0.5px dashed #CCCCCC"
        },
        frame: {
            paddingBottom: "19px",
            display: "flex"
        }
    });
    
    public render(): React.ReactNode {
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
            <div className={OverallTable.classNames.frame}>
                <div className={OverallTable.classNames.groupCol}>
                    <div className={OverallTable.classNames.groupLabel}>{/*this.props.binGroup*/}</div>
                    <div className={OverallTable.classNames.flexCol}>
                        <div className={OverallTable.classNames.binBox}>
                                <div className={OverallTable.classNames.binTitle}><b>{localization.Report.overallLabel}</b></div>
                        </div>
                        {this.props.binLabels.map((label, index) => {
                            if (this.props.expandAttributes) return (<div className={OverallTable.classNames.binBox} key={index}>
                                <div className={OverallTable.classNames.binTitle}>{label}</div>
                                <Stack horizontal>
                                    {minIndexes.includes(index) && <div className={OverallTable.classNames.minMaxLabel}>{localization.Report.minTag}</div>}
                                    {maxIndexes.includes(index) && <div className={OverallTable.classNames.minMaxLabel}>{localization.Report.maxTag}</div>}
                                </Stack>
                            </div>)
                        })}
                    </div>
                </div>
                {this.props.metricLabels.map((metric, index) => {
                    return (
                        <div className={OverallTable.classNames.metricCol}>
                        <div className={OverallTable.classNames.metricLabel}>{metric}</div>
                        <div className={OverallTable.classNames.flexCol}>
                            <div className={OverallTable.classNames.metricBox}>{this.props.overallMetrics[index]}</div>
                            {this.props.formattedBinValues[index].map((value, index) => {
                                if (this.props.expandAttributes) return (
                                <div className={OverallTable.classNames.metricBox} key={index}>
                                    {value !== undefined ? value : 'empty'}
                                </div>);
                            })}
                        </div>
                    </div>                        
                    )
                })}
            </div>
        );
    }
}