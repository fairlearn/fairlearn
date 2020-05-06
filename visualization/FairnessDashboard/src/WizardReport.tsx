import React from "react";
import ReactModal from "react-modal";
import { Stack, StackItem } from "office-ui-fabric-react/lib/Stack";
import { IModelComparisonProps } from "./Controls/ModelComparisonChart";
import { Text } from "office-ui-fabric-react/lib/Text";
import { localization } from "./Localization/localization";
import { Separator } from "office-ui-fabric-react/lib/Separator";
import { Dropdown, DropdownMenuItemType, IDropdownStyles, IDropdownOption } from 'office-ui-fabric-react/lib/Dropdown';
import { Spinner, SpinnerSize } from 'office-ui-fabric-react/lib/Spinner';
import { mergeStyleSets } from "@uifabric/styling";
import _ from "lodash";
import { ParityModes } from "./ParityMetrics";
import { IPlotlyProperty, AccessibleChart } from "mlchartlib";
import { ActionButton } from "office-ui-fabric-react/lib/Button";
import { SummaryTable } from "./Controls/SummaryTable";
import { OverallTable } from "./Controls/OverallTable";
import { PredictionTypes, IMetricResponse } from "./IFairnessProps";
import { AccuracyOptions } from "./AccuracyMetrics";
import { NONAME } from "dns";
import { ChartColors } from "./ChartColors";
import { withSlots } from "office-ui-fabric-react/lib/Foundation";

interface IMetrics {
    globalAccuracy: number;
    binnedAccuracy: number[];
    accuracyDisparity: number;
    globalOutcome: number;
    outcomeDisparity: number;
    binnedOutcome: number[];
    // Optional, based on model type
    binnedOverprediction?: number[];
    binnedUnderprediction?: number[];
    // different length, raw unbinned errors and predictions
    errors?: number[];
    predictions?: number[];
}

export interface IState {
    metrics?: IMetrics;
    featureKey?: string;
    showModalHelp?: boolean;
    expandAttributes: boolean;
}

export interface IReportProps extends IModelComparisonProps {
    selectedModelIndex: number;
}

export class WizardReport extends React.PureComponent<IReportProps, IState> {
    private static separatorStyle = {
        root: [{
          selectors: {
            '::after': {
              backgroundColor: 'darkgrey',
            },
          }
        }]
    };

    private static readonly classNames = mergeStyleSets({
        spinner: {
            margin: "auto",
            padding: "40px"
        },
        header: {
            padding: "0px 90px 20px 90px",
            backgroundColor: "#222222"
        },
        multimodelButton: {
            marginTop: "20px",
            padding: 0,
            color: "#ffffff",
            fontSize: "12px",
            fontWeight: "400"
        },
        headerTitle: {
            paddingTop: "10px",
            color: "#ffffff",
            fontSize: "30px",
            lineHeight: "43px",
            fontWeight: "300"
        },
        headerBanner: {
            display: "flex"
        },
        headerOptions: {
            backgroundColor: "#222222",
        },
        bannerWrapper: {
            width: "100%",
            paddingTop: "18px",
            paddingBottom: "15px",
            display: "inline-flex",
            flexDirection: "row",
            justifyContent: "space-between"
        },
        editButton: {
            color: "#ffffff",
            fontSize: "12px",
            lineHeight: "20px",
            fontWeight: "400"
        },
        metricText: {
            color: "#333333",
            fontSize: "36px",
            lineHeight: "44px",
            fontWeight: "100",
            paddingRight: "12px"
        },
        firstMetricLabel: {
            color: "#333333",
            fontSize: "12px",
            lineHeight: "16px",
            fontWeight: "400",
            padding: "8px 12px 0 12px",
            maxWidth: "120px",
            borderRight: "1px solid #CCCCCC",
            marginRight: "20px"
        },
        metricLabel: {
            color: "#333333",
            fontSize: "12px",
            lineHeight: "16px",
            fontWeight: "400",
            paddingTop: "8px",
            maxWidth: "130px"
        },
        expandAttributes: {
            color: "#333333",
            fontSize: "12px",
            lineHeight: "12px",
            fontWeight: "500",
            height: "26px",
            paddingLeft: "10px",
            marginLeft: "80px"
        },
        overallArea: {
            display: "flex",
            flexDirection: "row",
            padding: "20px 0 30px 90px",
            backgroundColor: 'white'
        },
        presentationArea: {
            display: "flex",
            flexDirection: "row",
            padding: "20px 0 30px 90px",
            backgroundColor: 'white'
        },
        chartWrapper: {
            flex: "1 0 40%",
            display: "flex",
            flexDirection: "column"
        },
        chartBody: {
            flex: 1
        },
        chartHeader: {
            height: "23px",
            paddingLeft: "10px",
            color: "#333333",
            fontSize: "12px",
            lineHeight: "12px",
            fontWeight: "500"
        },
        dropDown: {
            margin: "10px 0px",
            display: "inline-block"
        },
        mainRight: {
            minWidth: "200px",
            paddingLeft: "35px",
            flexBasis: "300px",
            flexShrink: 1
        },
        rightTitle: {
            color: "#333333",
            fontSize: "12px",
            lineHeight: "16px",
            fontWeight: "500",
            paddingBottom: "11px",
            borderBottom: "1px solid #CCCCCC"
        },
        rightText: {
            padding: "16px 15px 30px 0",
            color: "#333333",
            fontSize: "15px",
            lineHeight: "18px",
            fontWeight: "400"
            // borderBottom: "0.5px dashed #CCCCCC"
        },
        insights: {
            // textTransform: "uppercase",
            color: "#333333",
            fontSize: "15px",
            lineHeight: "16px",
            fontWeight: "600",
            padding: "18px 0",
        },
        insightsText: {
            color: "#333333",
            fontSize: "15px",
            lineHeight: "16px",
            fontWeight: "400",
            paddingBottom: "18px",
            paddingRight: "15px"
            // borderBottom: "1px solid #CCCCCC"
        },
        downloadReport: {
            color: "#333333",
            fontSize: "15px",
            lineHeight: "16px",
            fontWeight: "400",
            paddingTop: "20px",
            paddingBottom: "20px",
            paddingLeft: "60px",
            border: "1px solid #CCCCCC"
        },
        tableWrapper: {
            paddingBottom: "20px"
        },
        textRow: {
            float: "left",
            display: "flex",
            flexDirection: "row",
            alignItems: "center",
            paddingBottom: "7px",
            paddingLeft: "50px",
            paddingRight: "50px"
        },
        infoButton: {
            float: "left",
            width: "15px",
            height: "15px",
            textAlign: "center",
            fontFamily: "Arial",
            fontSize: "12px",
            lineHeight: "15px",
            fontWeight: "400",
            borderRadius: "50%",
            border: "1px solid",
            marginTop: "3px",
            marginRight: "3px",
            marginLeft: "250px"
        },
        closeButton: {
            color: "#FFFFFF",
            float: "right",
            fontFamily: "Arial",
            fontSize: "20px",
            lineHeight: "20px",
            fontWeight: "400",
            paddingLeft: "20px"
        },
        doneButton: {
            color: "#FFFFFF",
            borderRadius: "5px",
            background: "#5A53FF",
            padding: "5px 15px",
            selectors: {
                '&:hover': { color: "#ffffff" }
            }
        },
        equalizedOdds: {
            float: "left",
            fontWeight: "600",
            paddingTop: "30px",
            paddingLeft: "90px"
        },
        howTo: {
            paddingTop: "20px",
            paddingLeft: "100px"
        },
        colorBlock: {
            width: "15px",
            height: "15px",
            marginRight: "9px"
        },
        modalContentHelp: {
            float: 'left',
            paddingTop: '10px',
            paddingRight: '20px',
            wordWrap: "break-word",
            width: "300px",
            textAlign: "center"
        },
        multimodelSection: {
            display: "flex",
            flexDirection:"row"
        },
        modelLabel: {
            alignSelf: "center",
            color: "#ffffff",
            fontSize: "26px",
            fontWeight: "400",
            paddingTop: "10px",
            paddingBottom: "10px"
        }
    });

    private static barPlotlyProps: IPlotlyProperty = {
        config: { displaylogo: false, responsive: true, modeBarButtonsToRemove: ['toggleSpikelines', 'hoverClosestCartesian', 'hoverCompareCartesian', 'zoom2d', 'pan2d', 'select2d', 'lasso2d', 'zoomIn2d', 'zoomOut2d', 'autoScale2d', 'resetScale2d'] },
        data: [
            {
                orientation: 'h',
                type: 'bar'
            } as any
        ],
        layout: {
            autosize: true,
            barmode: 'relative',
            font: {
                size: 10
            },
            margin: {
                t: 4,
                l: 0,
                r: 0,
                b: 20
            },
            showlegend: false,
            hovermode: 'closest',
            plot_bgcolor: "#FAFAFA",
            xaxis: {
                fixedrange: true,
                autorange: true,
                mirror: true,
                linecolor: '#CCCCCC',
                linewidth: 1,
            },
            yaxis: {
                fixedrange: true,
                showticklabels: false,
                showgrid: true,
                dtick: 1,
                tick0: 0.5,
                gridcolor: '#CCCCCC',
                gridwidth: 1,
                autorange: "reversed"
            }
        } as any
    };

    render(): React.ReactNode {
        const dropdownStyles: Partial<IDropdownStyles> = {
            dropdown: { width: 180,selectors: {
                ':focus .ms-Dropdown-title': {
                    color: "#333333",
                    backgroundColor: "#f3f2f1",
                },
                ':hover .ms-Dropdown-title': {
                    color: "#333333",
                    backgroundColor: "#f3f2f1",
                }
            }},
            label: { color: "#ffffff" },
            title: { color: "#ffffff", backgroundColor: "#353535", borderWidth: "0px", borderRadius: "5px", selectors: {
                ':hover': {
                    color: "#333333",
                    backgroundColor: "#f3f2f1"
                }
            }},
            caretDown: { color: "#ffffff" },
            dropdownItem: { color: "#ffffff", backgroundColor: "#353535" }
        };
        const modalStyles = {
            content : {
                top                   : '50%',
                left                  : '50%',
                right                 : 'auto',
                bottom                : 'auto',
                marginRight           : '-50%',
                paddingTop            : '5px',
                paddingRight          : '10px',
                paddingBottom         : '10px',
                transform             : 'translate(-50%, -50%)',
                fontFamily: "Segoe UI",
                color: "#FFFFFF",
                borderRadius: "5px",
                backgroundColor: "#222222",
                boxShadow: "0px 10px 15px rgba(0, 0, 0, 0.1)"
            },
            overlay: {zIndex: 1000}
        };

        const featureOptions: IDropdownOption[] = this.props.dashboardContext.modelMetadata.featureNames.map(x => { return {key: x, text: x}});

        const alternateHeight = this.props.featureBinPickerProps.featureBins[this.props.featureBinPickerProps.selectedBinIndex].labelArray.length * 60 + 106;
        const areaHeights = Math.max(460, alternateHeight);

        const accuracyKey = this.props.accuracyPickerProps.selectedAccuracyKey;
        const outcomeKey = this.props.dashboardContext.modelMetadata.predictionType === PredictionTypes.binaryClassification ? "selection_rate" : "average";
        const outcomeMetric = AccuracyOptions[outcomeKey];

        const overpredicitonKey = "overprediction";
        const underpredictionKey = "underprediction";

        const accuracyPlot = _.cloneDeep(WizardReport.barPlotlyProps);
        const opportunityPlot = _.cloneDeep(WizardReport.barPlotlyProps);
        const nameIndex = this.props.dashboardContext.groupNames.map((unuxed, i) => i);
        let howToReadAccuracySection: React.ReactNode;
        let insightsAccuracySection: React.ReactNode;
        let howToReadOutcomesSection: React.ReactNode;
        let insightsOutcomesSection: React.ReactNode;
        let accuracyChartHeader: string = "";
        let opportunityChartHeader: string = "";

        var mainChart;
        if (!this.state || !this.state.metrics) {
            this.loadData();
            mainChart = <Spinner className={WizardReport.classNames.spinner} size={SpinnerSize.large} label={localization.calculating}/>;
        }
        else {
            if (this.props.dashboardContext.modelMetadata.predictionType === PredictionTypes.binaryClassification) {
                accuracyPlot.data = [
                    {
                        x: this.state.metrics.binnedOverprediction,
                        y: nameIndex,
                        text: this.state.metrics.binnedOverprediction.map(num => this.formatNumbers((num as number), "accuracy_score", false, 2)),
                        name: localization.Metrics.overprediction,
                        width: 0.5,
                        color: ChartColors[0],
                        orientation: 'h',
                        type: 'bar',
                        textposition: 'auto',
                        hoverinfo: "skip"
                    } as any, {
                        x: this.state.metrics.binnedUnderprediction.map(x => -1 * x),
                        y: nameIndex,
                        text: this.state.metrics.binnedUnderprediction.map(num => this.formatNumbers((num as number), "accuracy_score", false, 2)),
                        name: localization.Metrics.underprediction,
                        width: 0.5,
                        color: ChartColors[1],
                        orientation: 'h',
                        type: 'bar',
                        textposition: 'auto',
                        hoverinfo: "skip"
                    }
                ];
                accuracyPlot.layout.annotations = [
                    {
                        text: localization.Report.underestimationError,
                        x: 0.02,
                        y: 1,
                        yref: 'paper', xref: 'paper',
                        showarrow: false,
                        font: {color:'#666666', size: 10}
                    },
                    {
                        text: localization.Report.overestimationError,
                        x: 0.98,
                        y: 1,
                        yref: 'paper', xref: 'paper',
                        showarrow: false,
                        font: {color:'#666666', size: 10}
                    }
                ];
                accuracyPlot.layout.xaxis.tickformat = ',.0%';
                opportunityPlot.data = [
                    {
                        x: this.state.metrics.binnedOutcome,
                        y: nameIndex,
                        text: this.state.metrics.binnedOutcome.map(num => this.formatNumbers((num as number), "selection_rate", false, 2)),
                        name: outcomeMetric.title,
                        color: ChartColors[0],
                        orientation: 'h',
                        type: 'bar',
                        textposition: 'auto',
                        hoverinfo: "skip"
                    } as any
                ];
                opportunityPlot.layout.xaxis.tickformat = ',.0%';
                howToReadAccuracySection = (<div>
                    <div className={WizardReport.classNames.textRow}>{localization.Report.classificationAccuracyHowToRead3}</div>
                    <div className={WizardReport.classNames.textRow}>{localization.Report.classificationAccuracyHowToRead2}</div>
                    <div className={WizardReport.classNames.textRow}>{localization.Report.classificationAccuracyHowToRead3}</div>
                </div>);
                howToReadOutcomesSection = (<div>
                    <div className={WizardReport.classNames.textRow}>{localization.Report.classificationOutcomesHowToRead}</div>
                </div>);
            } if (this.props.dashboardContext.modelMetadata.predictionType === PredictionTypes.probability) {
                accuracyPlot.data = [
                    {
                        x: this.state.metrics.binnedOverprediction,
                        y: nameIndex,
                        text: this.state.metrics.binnedOverprediction.map(num => this.formatNumbers((num as number), "overprediction", false, 2)),
                        name: localization.Metrics.overprediction,
                        width: 0.5,
                        color: ChartColors[0],
                        orientation: 'h',
                        type: 'bar',
                        textposition: 'auto',
                        hoverinfo: "skip"
                    } as any, {
                        x: this.state.metrics.binnedUnderprediction.map(x => -1 * x),
                        y: nameIndex,
                        text: this.state.metrics.binnedUnderprediction.map(num => this.formatNumbers((num as number), "underprediction", false, 2)),
                        name: localization.Metrics.underprediction,
                        width: 0.5,
                        color: ChartColors[1],
                        orientation: 'h',
                        type: 'bar',
                        textposition: 'auto',
                        hoverinfo: "skip"
                    }
                ];
                accuracyPlot.layout.annotations = [
                    {
                        text: localization.Report.underestimationError,
                        x: 0.1,
                        y: 1,
                        yref: 'paper', xref: 'paper',
                        showarrow: false,
                        font: {color:'#666666', size: 10}
                    },
                    {
                        text: localization.Report.overestimationError,
                        x: 0.9,
                        y: 1,
                        yref: 'paper', xref: 'paper',
                        showarrow: false,
                        font: {color:'#666666', size: 10}
                    }
                ];
                const opportunityText = this.state.metrics.predictions.map(val => {
                    return localization.formatString(localization.Report.tooltipPrediction, 
                        this.formatNumbers((val as number), "average", false, 3));
                });
                opportunityPlot.data = [
                    {
                        x: this.state.metrics.predictions,
                        y: this.props.dashboardContext.binVector,
                        text: opportunityText,
                        type: 'box',
                        color: ChartColors[0],
                        boxmean: true,
                        orientation: 'h',
                        boxpoints: 'all',
                        hoverinfo: 'text',
                        hoveron: "points",
                        jitter: 0.4,
                        pointpos: 0,
                    } as any
                ];
                howToReadAccuracySection = (<div>
                    <div className={WizardReport.classNames.textRow}>
                        <div className={WizardReport.classNames.colorBlock} style={{backgroundColor: ChartColors[0]}}/>
                        <div>{localization.Report.overestimationError}</div>
                    </div>
                    <div className={WizardReport.classNames.textRow}>
                        <div className={WizardReport.classNames.colorBlock} style={{backgroundColor: ChartColors[1]}}/>
                        <div>{localization.Report.underestimationError}</div>
                    </div>
                    <div className={WizardReport.classNames.textRow}>{localization.Report.probabilityAccuracyHowToRead1}</div>
                    <div className={WizardReport.classNames.textRow}>{localization.Report.probabilityAccuracyHowToRead2}</div>
                    <div className={WizardReport.classNames.textRow}>{localization.Report.probabilityAccuracyHowToRead3}</div>
                </div>);
                howToReadOutcomesSection = (<div>
                    <div className={WizardReport.classNames.textRow}>{localization.Report.regressionOutcomesHowToRead}</div>
                </div>);
                opportunityChartHeader = localization.Report.distributionOfPredictions;
            } if (this.props.dashboardContext.modelMetadata.predictionType === PredictionTypes.regression) {
                const opportunityText = this.state.metrics.predictions.map(val => {
                    return localization.formatString(localization.Report.tooltipPrediction, val);
                });
                const accuracyText = this.state.metrics.predictions.map((val, index) => {
                    return `${localization.formatString(
                            localization.Report.tooltipError, 
                            this.formatNumbers((this.state.metrics.errors[index] as number), "average", false, 3))
                        }<br>${localization.formatString(
                            localization.Report.tooltipPrediction, 
                            this.formatNumbers((val as number), "average", false, 3))}`;
                });
                accuracyPlot.data = [
                    {
                        x: this.state.metrics.errors,
                        y: this.props.dashboardContext.binVector,
                        text: accuracyText,
                        type: 'box',
                        color: ChartColors[0],
                        orientation: 'h',
                        boxmean: true,
                        hoveron: "points",
                        hoverinfo: 'text',
                        boxpoints: 'all',
                        jitter: 0.4,
                        pointpos: 0,
                    } as any
                ];
                opportunityPlot.data = [
                    {
                        x: this.state.metrics.predictions,
                        y: this.props.dashboardContext.binVector,
                        text: opportunityText,
                        type: 'box',
                        color: ChartColors[0],
                        boxmean: true,
                        orientation: 'h',
                        hoveron: "points",
                        boxpoints: 'all',
                        hoverinfo: 'text',
                        jitter: 0.4,
                        pointpos: 0,
                    } as any
                ];
                howToReadAccuracySection = (<div>
                    <div className={WizardReport.classNames.textRow}>{localization.Report.regressionAccuracyHowToRead}</div>
                </div>);
                howToReadOutcomesSection = (<div>
                    <div className={WizardReport.classNames.textRow}>{localization.Report.regressionOutcomesHowToRead}</div>
                </div>);
                opportunityChartHeader = localization.Report.distributionOfPredictions;
                accuracyChartHeader = localization.Report.distributionOfErrors;
            }
            
            const globalAccuracyString = this.formatNumbers(this.state.metrics.globalAccuracy, accuracyKey);
            const disparityAccuracyString = this.formatNumbers(this.state.metrics.accuracyDisparity, accuracyKey);
            
            const globalOutcomeString = this.formatNumbers(this.state.metrics.globalOutcome, outcomeKey);
            const disparityOutcomeString = this.formatNumbers(this.state.metrics.outcomeDisparity, outcomeKey);

            const formattedBinAccuracyValues = this.state.metrics.binnedAccuracy.map(value => 
                this.formatNumbers(value, accuracyKey));
            const formattedBinOutcomeValues = this.state.metrics.binnedOutcome.map(value => 
                this.formatNumbers(value, outcomeKey));
            const formattedBinOverPredictionValues = this.state.metrics.binnedOverprediction.map(value => 
                this.formatNumbers(value, overpredicitonKey));
            const formattedBinUnderPredictionValues = this.state.metrics.binnedUnderprediction.map(value => 
                this.formatNumbers(value, underpredictionKey));

            const overallMetrics = [globalAccuracyString, globalOutcomeString, disparityAccuracyString, disparityOutcomeString];
            const formattedBinValues = [formattedBinAccuracyValues, formattedBinOutcomeValues, formattedBinOverPredictionValues, formattedBinUnderPredictionValues];
            const metricLabels = [AccuracyOptions[accuracyKey].title, AccuracyOptions[outcomeKey].title, AccuracyOptions[overpredicitonKey].title, AccuracyOptions[underpredictionKey].title];

            mainChart = 
                    <div>
                        <div className={WizardReport.classNames.overallArea} style={{height: !this.state.expandAttributes && "150px" || this.state.expandAttributes && `${areaHeights/2}px` }}>
                            <OverallTable
                                binGroup={this.props.dashboardContext.modelMetadata.featureNames[this.props.featureBinPickerProps.selectedBinIndex]}
                                binLabels={this.props.dashboardContext.groupNames}
                                formattedBinValues={formattedBinValues}
                                metricLabels={metricLabels}
                                overallMetrics={overallMetrics}
                                expandAttributes={this.state.expandAttributes}
                                binValues={this.state.metrics.binnedAccuracy}/>
                        </div>
                        <div className={WizardReport.classNames.expandAttributes} onClick={this.expandAttributes}>{this.state.expandAttributes && localization.Report.collapseSensitiveAttributes || !this.state.expandAttributes && localization.Report.expandSensitiveAttributes}</div>
                        <div className={WizardReport.classNames.equalizedOdds}>{localization.Report.equalizedOddsDisparity}</div>
                        <div className={WizardReport.classNames.howTo}>
                                <ActionButton onClick={this.handleOpenModalHelp}><div className={WizardReport.classNames.infoButton}>i</div>{localization.ModelComparison.howToRead}</ActionButton>
                                <ReactModal
                                    style={modalStyles}
                                    appElement={document.getElementById('app') as HTMLElement}
                                    isOpen={this.state.showModalHelp}
                                    contentLabel="Minimal Modal Example"
                                    >
                                    {/* <ActionButton className={WizardReport.classNames.closeButton} onClick={this.handleCloseModalHelp}>x</ActionButton> */}
                                    <p className={WizardReport.classNames.modalContentHelp}>{localization.Report.classificationAccuracyHowToRead1}<br/><br/>{localization.Report.classificationAccuracyHowToRead2}<br/><br/>{localization.Report.classificationAccuracyHowToRead3}<br /><br /><ActionButton className={WizardReport.classNames.doneButton} onClick={this.handleCloseModalHelp}>Done</ActionButton></p>
                                </ReactModal>
                        </div>
                        <div className={WizardReport.classNames.presentationArea} style={{height: `${areaHeights}px`}}>
                            <SummaryTable 
                                binGroup={this.props.dashboardContext.modelMetadata.featureNames[this.props.featureBinPickerProps.selectedBinIndex]}
                                binLabels={this.props.dashboardContext.groupNames}
                                formattedBinValues={formattedBinAccuracyValues}
                                metricLabel={AccuracyOptions[accuracyKey].title}
                                binValues={this.state.metrics.binnedAccuracy}/>
                            <div className={WizardReport.classNames.chartWrapper}>
                                <div className={WizardReport.classNames.chartHeader}>{accuracyChartHeader}</div>
                                <div className={WizardReport.classNames.chartBody}>
                                    <AccessibleChart
                                        plotlyProps={accuracyPlot}
                                        sharedSelectionContext={undefined}
                                        theme={undefined}
                                    />
                                </div>
                            </div>
                            <div className={WizardReport.classNames.mainRight}>
                                {/* <div className={WizardReport.classNames.rightTitle}>{localization.ModelComparison.howToRead}</div> */}
                                {/* <div className={WizardReport.classNames.rightText}>{howToReadAccuracySection}</div> */}
                                <div className={WizardReport.classNames.insights}>{localization.ModelComparison.insights}</div>
                                <div className={WizardReport.classNames.insightsText}>{localization.loremIpsum}</div>
                                <div className={WizardReport.classNames.downloadReport}>{localization.ModelComparison.downloadReport}</div>
                            </div>
                     </div>
                    <div className={WizardReport.classNames.textRow}>
                        <div className={WizardReport.classNames.colorBlock} style={{backgroundColor: ChartColors[1]}}/>
                        <div>
                            <div>{localization.Report.underestimationError}</div>
                            <div>{localization.Report.underpredictionExplanation}</div>
                        </div>
                    </div>
                    <div className={WizardReport.classNames.textRow}>
                        <div className={WizardReport.classNames.colorBlock} style={{backgroundColor: ChartColors[0]}}/>
                        <div>
                            <div>{localization.Report.overestimationError}</div>
                            <div>{localization.Report.overpredictionExplanation}</div>
                        </div>
                    </div>
                </div>
        }

        return (<div style={{height: "100%", overflowY:"auto"}}>
            <div className={WizardReport.classNames.header}>
                {this.props.modelCount > 1 &&
                    <div className={WizardReport.classNames.multimodelSection}>
                        <ActionButton
                            className={WizardReport.classNames.multimodelButton}
                            iconProps={{iconName: "ChevronLeft"}}
                            onClick={this.clearModelSelection}>
                            {localization.Report.backToComparisons}
                        </ActionButton>
                    </div>}
                    <div className={WizardReport.classNames.modelLabel}>
                        {localization.Report.assessmentResults} <b>{this.props.dashboardContext.modelNames[this.props.selectedModelIndex]}</b>
                    </div>
                    <div className={WizardReport.classNames.headerOptions}>
                        <Dropdown
                            className={WizardReport.classNames.dropDown}
                            // label="Feature"
                            defaultSelectedKey={this.props.dashboardContext.modelMetadata.featureNames[this.props.featureBinPickerProps.selectedBinIndex]}
                            options={featureOptions}
                            disabled={false}
                            onChange={this.featureChanged}
                            styles={dropdownStyles}
                        />
                    </div>
            </div>
            {mainChart}
        </div>);
    }

    private readonly formatNumbers = (value: number, key: string, isRatio: boolean = false, sigDigits: number = 3): string => {
        if (value === null || value === undefined || value === NaN) {
            return NaN.toString();
        }
        const styleObject = {maximumSignificantDigits: sigDigits};
        if (AccuracyOptions[key].isPercentage && !isRatio) {
            (styleObject as any).style = "percent";
        }
        return value.toLocaleString(undefined, styleObject);
    }

    private readonly clearModelSelection = (): void => {
        this.props.selections.onSelect([]);
    }

    private readonly expandAttributes = (): void => {
        this.setState({ expandAttributes: !this.state.expandAttributes });
    }

    private readonly onEditConfigs = (): void => {
        if (this.props.modelCount > 1) {
            this.props.selections.onSelect([]);
        }
        this.props.onEditConfigs();
    }

    private readonly handleOpenModalHelp = (event): void => {
        this.setState({ showModalHelp: true });
    }

    private readonly handleCloseModalHelp = (event): void => {
        this.setState({ showModalHelp: false });
    }

    private readonly featureChanged = (ev: React.FormEvent<HTMLInputElement>, option: IDropdownOption): void => {
        const featureKey = option.key.toString();
        if (this.state.featureKey !== featureKey) {
            this.props.featureBinPickerProps.selectedBinIndex = this.props.dashboardContext.modelMetadata.featureNames.indexOf(featureKey);
            this.setState({featureKey: featureKey, metrics: undefined});
        }
    }

    private async loadData(): Promise<void> {
        try {
            let binnedFNR: number[];
            let binnedFPR: number[];
            let binnedOverprediction: number[];
            let binnedUnderprediction: number[];
            let predictions: number[];
            let errors: number[];
            let outcomes: IMetricResponse;
            let outcomeDisparity: number;
            const accuracy = (await this.props.metricsCache.getMetric(
                this.props.dashboardContext.binVector,
                this.props.featureBinPickerProps.selectedBinIndex, 
                this.props.selectedModelIndex,
                this.props.accuracyPickerProps.selectedAccuracyKey));
            const accuracyDisparity = await this.props.metricsCache.getDisparityMetric(
                this.props.dashboardContext.binVector,
                this.props.featureBinPickerProps.selectedBinIndex, 
                this.props.selectedModelIndex,
                this.props.accuracyPickerProps.selectedAccuracyKey,
                ParityModes.difference);
            if (this.props.dashboardContext.modelMetadata.predictionType === PredictionTypes.binaryClassification) {
                binnedUnderprediction = (await this.props.metricsCache.getMetric(
                    this.props.dashboardContext.binVector,
                    this.props.featureBinPickerProps.selectedBinIndex, 
                    this.props.selectedModelIndex,
                    "underprediction")).bins;
                binnedOverprediction = (await this.props.metricsCache.getMetric(
                    this.props.dashboardContext.binVector,
                    this.props.featureBinPickerProps.selectedBinIndex, 
                    this.props.selectedModelIndex,
                    "overprediction")).bins;
                outcomes = await this.props.metricsCache.getMetric(
                    this.props.dashboardContext.binVector,
                    this.props.featureBinPickerProps.selectedBinIndex, 
                    this.props.selectedModelIndex,
                    "selection_rate");
                outcomeDisparity = await this.props.metricsCache.getDisparityMetric(
                    this.props.dashboardContext.binVector,
                    this.props.featureBinPickerProps.selectedBinIndex, 
                    this.props.selectedModelIndex,
                    "selection_rate",
                    ParityModes.difference);
            } if (this.props.dashboardContext.modelMetadata.predictionType === PredictionTypes.probability) {
                predictions = this.props.dashboardContext.predictions[this.props.selectedModelIndex];
                binnedOverprediction = (await this.props.metricsCache.getMetric(
                    this.props.dashboardContext.binVector,
                    this.props.featureBinPickerProps.selectedBinIndex, 
                    this.props.selectedModelIndex,
                    "overprediction")).bins;
                binnedUnderprediction = (await this.props.metricsCache.getMetric(
                    this.props.dashboardContext.binVector,
                    this.props.featureBinPickerProps.selectedBinIndex, 
                    this.props.selectedModelIndex,
                    "underprediction")).bins;
                outcomes = await this.props.metricsCache.getMetric(
                    this.props.dashboardContext.binVector,
                    this.props.featureBinPickerProps.selectedBinIndex, 
                    this.props.selectedModelIndex,
                    "average");
                outcomeDisparity = await this.props.metricsCache.getDisparityMetric(
                    this.props.dashboardContext.binVector,
                    this.props.featureBinPickerProps.selectedBinIndex, 
                    this.props.selectedModelIndex,
                    "average",
                    ParityModes.difference);
            } if (this.props.dashboardContext.modelMetadata.predictionType === PredictionTypes.regression) {
                predictions = this.props.dashboardContext.predictions[this.props.selectedModelIndex];
                errors = predictions.map((predicted, index) => {
                    return predicted - this.props.dashboardContext.trueY[index];
                });
                outcomes = await this.props.metricsCache.getMetric(
                    this.props.dashboardContext.binVector,
                    this.props.featureBinPickerProps.selectedBinIndex, 
                    this.props.selectedModelIndex,
                    "average");
                outcomeDisparity = await this.props.metricsCache.getDisparityMetric(
                    this.props.dashboardContext.binVector,
                    this.props.featureBinPickerProps.selectedBinIndex, 
                    this.props.selectedModelIndex,
                    "average",
                    ParityModes.difference);
            }
            this.setState({
                metrics: {
                    globalAccuracy: accuracy.global,
                    binnedAccuracy: accuracy.bins,
                    accuracyDisparity,
                    globalOutcome: outcomes.global,
                    binnedOutcome: outcomes.bins,
                    outcomeDisparity,
                    predictions,
                    errors,
                    binnedOverprediction,
                    binnedUnderprediction
                }
            });
        } catch {
            // todo;
        }
    }
}