import React from "react";
import { Stack, StackItem } from "office-ui-fabric-react/lib/Stack";
import { IModelComparisonProps } from "./Controls/ModelComparisonChart";
import { Text } from "office-ui-fabric-react/lib/Text";
import { localization } from "./Localization/localization";
import { Separator } from "office-ui-fabric-react/lib/Separator";
import { Spinner, SpinnerSize } from 'office-ui-fabric-react/lib/Spinner';
import { mergeStyleSets } from "@uifabric/styling";
import _ from "lodash";
import { ParityModes } from "./ParityMetrics";
import { IPlotlyProperty, AccessibleChart } from "mlchartlib";
import { ActionButton } from "office-ui-fabric-react/lib/Button";
import { SummaryTable } from "./Controls/SummaryTable";

interface IMetrics {
    globalAccuracy: number;
    accuracyDisparity: number;
    binnedFNR: number[];
    binnedFPR: number[];
    globalOutcome: number;
    outcomeDisparity: number;
    binnedOutcome: number[];
}

export interface IState {
    metrics?: IMetrics;
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
            fontFamily: `"Segoe UI", "Segoe UI Web (West European)", "Segoe UI", -apple-system, BlinkMacSystemFont, Roboto, "Helvetica Neue", sans-serif`,
            padding: "40px"
        },
        percentageText: {
            alignSelf: "center",
            fontSize: "55px"
        },
        percentageLabel: {
            alignSelf: "center"
        }
    });

    private static barPlotlyProps: IPlotlyProperty = {
        config: { displaylogo: false, responsive: true, modeBarButtonsToRemove: ['toggleSpikelines', 'hoverClosestCartesian', 'hoverCompareCartesian', 'lasso2d', 'select2d'] },
        data: [
            {
                orientation: 'h',
                type: 'bar'
            }
        ],
        layout: {
            autosize: true,
            barmode: 'relative',
            font: {
                size: 10
            },
            margin: {
                t: 10,
                l: 10
            },
            showlegend: false,
            hovermode: 'closest',
            xaxis: {
                automargin: true
            },
            yaxis: {
                automargin: true,
                showticklabels: false
            },
        } as any
    };

    render(): React.ReactNode {
        if (!this.state || !this.state.metrics) {
            this.loadData();
            return (
                <Spinner className={WizardReport.classNames.spinner} size={SpinnerSize.large} label={localization.calculating}/>
            );
        }

        const accuracyPlot = _.cloneDeep(WizardReport.barPlotlyProps);
        accuracyPlot.data = [
            {
                x: this.state.metrics.binnedFPR,
                y: this.props.dashboardContext.groupNames,
                text: this.state.metrics.binnedFPR.map(num => (num as number).toFixed(3)),
                orientation: 'h',
                type: 'bar',
                textposition: 'auto'
            } as any, {
                x: this.state.metrics.binnedFNR.map(x => -1 * x),
                y: this.props.dashboardContext.groupNames,
                text: this.state.metrics.binnedFNR.map(num => (num as number).toFixed(3)),
                orientation: 'h',
                type: 'bar',
                textposition: 'auto'
            }
        ];

        const opportunityPlot = _.cloneDeep(WizardReport.barPlotlyProps);
        opportunityPlot.data = [
            {
                x: this.state.metrics.binnedOutcome,
                y: this.props.dashboardContext.groupNames,
                text: this.state.metrics.binnedOutcome.map(num => (num as number).toFixed(3)),
                orientation: 'h',
                type: 'bar',
                textposition: 'auto'
            } as any
        ];
        
        return (<div style={{height: "100%", overflowY:"auto"}}>
            <div style={{backgroundColor: "#EBEBEB", padding: "10px 30px"}}>
                {this.props.modelCount > 0 && <StackItem>
                    <ActionButton iconProps={{iconName: "ChromeBack"}} onClick={this.clearModelSelection}>{localization.Report.backToComparisons}</ActionButton>
                </StackItem>}
                <StackItem>
                    <Text variant={"xxLarge"}>{localization.Report.title}</Text>
                </StackItem>
                <Stack.Item styles={{root: {height: "100px"}}}>
                    <Stack horizontal horizontalAlign="space-between" styles={{root: {height: "100%"}}}>
                        <StackItem>
                            <Stack horizontal styles={{root: {height: "100%"}}}>
                                <Text className={WizardReport.classNames.percentageText}>{this.state.metrics.globalAccuracy.toLocaleString(undefined, {style: "percent", maximumFractionDigits: 1})}</Text>
                                <Text variant={"medium"} className={WizardReport.classNames.percentageLabel}>{localization.Report.globalAccuracyText}</Text>
                                <Separator vertical styles={WizardReport.separatorStyle} />
                                <Text className={WizardReport.classNames.percentageText}>{this.state.metrics.accuracyDisparity.toLocaleString(undefined, {style: "percent", maximumFractionDigits: 1})}</Text>
                                <Text variant={"medium"} className={WizardReport.classNames.percentageLabel}>{localization.Report.accuracyDisparityText}</Text>
                            </Stack>
                        </StackItem>
                        <ActionButton iconProps={{iconName: "Edit"}} onClick={this.props.onEditConfigs}>{localization.Report.editConfiguration}</ActionButton>
                    </Stack>
                </Stack.Item>
            </div>
            <div style={{padding: "10px 30px", height:"400px"}}>
                <Stack horizontal styles={{root: {height: "100%"}}}>
                    <SummaryTable 
                        binLabels={this.props.dashboardContext.groupNames}
                        binValues={this.state.metrics.binnedFPR}/>
                    <Stack.Item styles={{root: {width: "55%"}}}>
                        <AccessibleChart
                            plotlyProps={accuracyPlot}
                            sharedSelectionContext={undefined}
                            theme={undefined}
                        />
                    </Stack.Item>
                </Stack>
            </div>
            <div style={{backgroundColor: "#EBEBEB", padding: "10px 30px"}}>
                <Stack.Item styles={{root: {height: "100px"}}}>
                    <Stack horizontal styles={{root: {height: "100%"}}}>
                        <Text className={WizardReport.classNames.percentageText}>{this.state.metrics.globalOutcome.toLocaleString(undefined, {style: "percent", maximumFractionDigits: 1})}</Text>
                        <Text variant={"medium"} className={WizardReport.classNames.percentageLabel}>{localization.Report.globalOutcomeText}</Text>
                        <Separator vertical styles={WizardReport.separatorStyle} />
                        <Text className={WizardReport.classNames.percentageText}>{this.state.metrics.outcomeDisparity.toLocaleString(undefined, {style: "percent", maximumFractionDigits: 1})}</Text>
                        <Text variant={"medium"} className={WizardReport.classNames.percentageLabel}>{localization.Report.outcomeDisparityText}</Text>
                    </Stack>
                </Stack.Item>
            </div>
            <div style={{padding: "10px 30px", height:"400px"}}>
                <Stack horizontal styles={{root: {height: "100%"}}}>
                    <SummaryTable 
                        binLabels={this.props.dashboardContext.groupNames}
                        binValues={this.state.metrics.binnedFPR}/>
                    <Stack.Item styles={{root: {width: "55%"}}}>
                        <AccessibleChart
                            plotlyProps={opportunityPlot}
                            sharedSelectionContext={undefined}
                            theme={undefined}
                        />
                    </Stack.Item>
                </Stack>
            </div>
        </div>);
    }

    private readonly clearModelSelection = (): void => {
        this.props.selections.onSelect([]);
    }

    private async loadData(): Promise<void> {
        try {
            const globalAccuracy = (await this.props.metricsCache.getMetric(
                this.props.dashboardContext.binVector,
                this.props.featureBinPickerProps.selectedBinIndex, 
                this.props.selectedModelIndex,
                this.props.accuracyPickerProps.selectedAccuracyKey)).global;
            const accuracyDisparity = await this.props.metricsCache.getDisparityMetric(
                this.props.dashboardContext.binVector,
                this.props.featureBinPickerProps.selectedBinIndex, 
                this.props.selectedModelIndex,
                this.props.accuracyPickerProps.selectedAccuracyKey,
                ParityModes.difference);
            const binnedFNR = (await this.props.metricsCache.getMetric(
                this.props.dashboardContext.binVector,
                this.props.featureBinPickerProps.selectedBinIndex, 
                this.props.selectedModelIndex,
                // TODO: use fnr et al when available
                // "fnr")).bins;
                this.props.accuracyPickerProps.selectedAccuracyKey)).bins;
            const binnedFPR = (await this.props.metricsCache.getMetric(
                this.props.dashboardContext.binVector,
                this.props.featureBinPickerProps.selectedBinIndex, 
                this.props.selectedModelIndex,
                // "fpr")).bins;
                this.props.accuracyPickerProps.selectedAccuracyKey)).bins;
            const outcomes = await this.props.metricsCache.getMetric(
                this.props.dashboardContext.binVector,
                this.props.featureBinPickerProps.selectedBinIndex, 
                this.props.selectedModelIndex,
                // "outcomes");
                this.props.accuracyPickerProps.selectedAccuracyKey);
            const outcomeDisparity = await this.props.metricsCache.getDisparityMetric(
                this.props.dashboardContext.binVector,
                this.props.featureBinPickerProps.selectedBinIndex, 
                this.props.selectedModelIndex,
                // "outcomes",
                this.props.accuracyPickerProps.selectedAccuracyKey,
                ParityModes.difference);
            this.setState({
                metrics: {
                    globalAccuracy,
                    accuracyDisparity,
                    binnedFNR,
                    binnedFPR,
                    globalOutcome: outcomes.global,
                    binnedOutcome: outcomes.bins,
                    outcomeDisparity
                }
            });
        } catch {
            // todo;
        }
    }
}