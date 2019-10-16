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
                t: 10
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
        accuracyPlot.layout.yaxis.title = this.props.dashboardContext.modelMetadata.featureNames[
            this.props.featureBinPickerProps.selectedBinIndex];
        accuracyPlot.layout.yaxis.showticklabels = true;

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

        opportunityPlot.layout.yaxis.title = this.props.dashboardContext.modelMetadata.featureNames[
            this.props.featureBinPickerProps.selectedBinIndex];
        opportunityPlot.layout.yaxis.showticklabels = true;
        
        return (<Stack styles={{root: {height: "100%"}}}>
            <StackItem styles={{root: {backgroundColor: "#EBEBEB", padding: "10px 30px"}}}>
                <StackItem>
                    <Text variant={"xxLarge"}>{localization.Report.title}</Text>
                </StackItem>
                <Stack.Item styles={{root: {height: "100px"}}}>
                    <Stack horizontal horizontalAlign="space-between" styles={{root: {height: "100%"}}}>
                        <StackItem>
                            <Stack horizontal styles={{root: {height: "100%"}}}>
                                <Text variant={"xxLarge"}>{this.state.metrics.globalAccuracy.toLocaleString(undefined, {style: "percent", maximumFractionDigits: 1})}</Text>
                                <Text variant={"medium"}>{localization.Report.globalAccuracyText}</Text>
                                <Separator vertical styles={WizardReport.separatorStyle} />
                                <Text variant={"xxLarge"}>{this.state.metrics.accuracyDisparity.toLocaleString(undefined, {style: "percent", maximumFractionDigits: 1})}</Text>
                                <Text variant={"medium"}>{localization.Report.accuracyDisparityText}</Text>
                            </Stack>
                        </StackItem>
                        <StackItem>{localization.Report.editConfiguration}</StackItem>
                    </Stack>
                </Stack.Item>
            </StackItem>
            <Stack.Item grow={2} styles={{root: {padding: "10px 30px"}}}>
                <Stack horizontal styles={{root: {height: "100%"}}}>
                    <Stack.Item styles={{root: {width: "55%"}}}>
                        <AccessibleChart
                            plotlyProps={accuracyPlot}
                            sharedSelectionContext={undefined}
                            theme={undefined}
                        />
                    </Stack.Item>
                    <Stack.Item>
                        table
                    </Stack.Item>
                    <Stack.Item grow={2}>
                        <AccessibleChart
                            plotlyProps={opportunityPlot}
                            sharedSelectionContext={undefined}
                            theme={undefined}
                        />
                    </Stack.Item>
                </Stack>
            </Stack.Item>
        </Stack>);
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