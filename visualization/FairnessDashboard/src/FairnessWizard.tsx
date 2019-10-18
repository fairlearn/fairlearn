import { IFairnessProps } from "./IFairnessProps";
import React from "react";
import { IFairnessContext, IFairnessModelMetadata } from "./IFairnessContext";
import { localization } from "./Localization/localization";
import _ from "lodash";
import { Pivot, PivotItem } from "office-ui-fabric-react/lib/Pivot";
import { Stack, StackItem } from "office-ui-fabric-react/lib/Stack";
import { SelectionContext, ICategoricalRange, IModelMetadata, ModelMetadata, RangeTypes } from "mlchartlib";
import { AccuracyOptions, IAccuracyOption } from "./AccuracyMetrics";
import { WizardReport } from "./WizardReport";
import { AccuracyTab } from "./Controls/AccuracyTab";
import { ParityTab } from "./Controls/ParityTab";
import { ParityOptions, IParityOption } from "./ParityMetrics";
import { MetricsCache } from "./MetricsCache";
import { ModelComparisonChart } from "./Controls/ModelComparisonChart";
import { FeatureTab } from "./Controls/FeatureTab";
import { IBinnedResponse } from "./IBinnedResponse";
import { Text } from "office-ui-fabric-react/lib/Text";
import { IntroTab } from "./Controls/IntroTab";

export interface IAccuracyPickerProps {
    accuracyOptions: IAccuracyOption[];
    selectedAccuracyKey: string;
    onAccuracyChange: (newKey: string) => void;
}

export interface IParityPickerProps {
    parityOptions: IAccuracyOption[];
    selectedParityKey: string;
    onParityChange: (newKey: string) => void;
}

export interface IFeatureBinPickerProps {
    featureBins: IBinnedResponse[];
    selectedBinIndex: number;
    onBinChange: (index: number) => void;
}

export interface IWizardState {
    activeTabKey: string;
    selectedModelId?: number;
    dashboardContext: IFairnessContext;
    accuracyMetrics: IAccuracyOption[];
    parityMetrics: IAccuracyOption[];
    selectedAccuracyKey: string;
    selectedParityKey: string;
    featureBins: IBinnedResponse[];
    selectedBinIndex: number;
    metricCache: MetricsCache;
}

const trueYPath = "trueY";
const indexPath = "index";

const introTabKey = "introTab";
const featureBinTabKey = "featureBinTab";
const accuracyTabKey = "accuracyTab";
const disparityTabKey = "disparityTab";
const reportTabKey = "reportTab"

const flights = {
    skipDisparity: true
}


export class FairnessWizard extends React.PureComponent<IFairnessProps, IWizardState> {
    private static buildInitialFairnessContext(props: IFairnessProps): IFairnessContext {
        const dataset = props.testData.map((row, rowIndex) => {
            const result = {...row};
            result[trueYPath] = props.trueY[rowIndex];
            result[indexPath] = rowIndex;
            return result;
        });
        return {
            dataset,
            predictions: props.predictedY,
            binVector: [],
            groupNames: [],
            modelMetadata: FairnessWizard.buildModelMetadata(props),
            modelNames: props.predictedY.map((unused, modelIndex) => `model ${modelIndex}`)
        };
    }   

    private static getClassLength(props: IFairnessProps): number {
        return _.uniq(props.trueY).length;
    }

    private static buildModelMetadata(props: IFairnessProps): IFairnessModelMetadata {
        let featureNames = props.dataSummary.featureNames;
        if (!featureNames) {
            let featureLength = 0;
            if (props.testData && props.testData[0] !== undefined) {
                featureLength = props.testData[0].length;
            }
            featureNames = ModelMetadata.buildIndexedNames(featureLength, localization.defaultFeatureNames);
        }
        const classNames = props.dataSummary.classNames || ModelMetadata.buildIndexedNames(FairnessWizard.getClassLength(props), localization.defaultClassNames);
        const featureIsCategorical = ModelMetadata.buildIsCategorical(featureNames.length, props.testData, props.dataSummary.categoricalMap);
        const featureRanges = ModelMetadata.buildFeatureRanges(props.testData, featureIsCategorical, props.dataSummary.categoricalMap);
        return {
            featureNames,
            featureNamesAbridged: featureNames,
            classNames,
            featureIsCategorical,
            featureRanges,
            predictionType: props.predictionType ||  "classes"
        };
    }

    private selections: SelectionContext;

    constructor(props: IFairnessProps) {
        super(props);
        const fairnessContext = FairnessWizard.buildInitialFairnessContext(props);

        this.selections = new SelectionContext("models", 1);
        this.selections.subscribe({selectionCallback: (strings: string[]) => {
            const numbers = strings.map(s => +s);
            this.setSelectedModel(numbers[0]);
        }});

        const featureBins = this.buildFeatureBins(fairnessContext.modelMetadata);
        fairnessContext.binVector = this.generateBinVectorForBin(featureBins[0], fairnessContext.dataset);
        fairnessContext.groupNames = this.generateStringLabelsForBins(featureBins[0], fairnessContext.modelMetadata);

        let accuracyMetrics = fairnessContext.modelMetadata.predictionType === "classes" ?
            this.props.supportedClassificationAccuracyKeys.map(key => AccuracyOptions[key]) :
            this.props.supportedRegressionAccuracyKeys.map(key => AccuracyOptions[key])
        accuracyMetrics.filter(metric => !!metric);

        this.state = {
            accuracyMetrics,
            selectedAccuracyKey: accuracyMetrics[0].key,
            parityMetrics: accuracyMetrics,
            selectedParityKey: accuracyMetrics[0].key,
            dashboardContext: fairnessContext,
            activeTabKey: introTabKey,
            featureBins,
            selectedBinIndex: 0,
            metricCache: new MetricsCache(
                featureBins.length,
                this.props.predictedY.length,
                this.props.requestMetrics)
        };
    }

    public render(): React.ReactNode {
        const accuracyPickerProps = {
            accuracyOptions: this.state.accuracyMetrics,
            selectedAccuracyKey: this.state.selectedAccuracyKey,
            onAccuracyChange: this.setAccuracyKey
        };
        const parityPickerProps = {
            parityOptions: this.state.parityMetrics,
            selectedParityKey: this.state.selectedParityKey,
            onParityChange: this.setParityKey
        };
        const featureBinPickerProps = {
            featureBins: this.state.featureBins,
            selectedBinIndex: this.state.selectedBinIndex,
            onBinChange: this.setBinIndex
        };
         return (
             <Stack style={{height: '800px'}}>
                <Stack horizontal horizontalAlign="space-between" verticalAlign="center" style={{
                     height: "35px",
                     backgroundColor: "#333",
                     color: "#CCC"
                     }}>
                    <Text variant={"mediumPlus"} style={{padding: "20px"}}>{localization.Header.title}</Text>
                    <Text variant={"medium"} style={{padding: "15px"}}>{localization.Header.documentation}</Text>
                </Stack>
                {(this.state.activeTabKey === introTabKey) &&
                    <StackItem grow={2}>
                        <IntroTab onNext={this.setTab.bind(this, featureBinTabKey)}/>
                    </StackItem>}
                 {(this.state.activeTabKey === featureBinTabKey ||
                   this.state.activeTabKey === accuracyTabKey ||
                   this.state.activeTabKey === disparityTabKey
                 ) &&
                    <Stack.Item grow={2}>
                        <Pivot
                            style={{
                                height: "100%",
                                display: "flex",
                                flexDirection: "column"
                            }}
                            styles={{
                                itemContainer: {
                                    height: "100%"
                                }
                            }}
                            selectedKey={this.state.activeTabKey}
                            onLinkClick={this.handleTabClick}>
                            <PivotItem headerText={"Protected Attributes"} itemKey={featureBinTabKey} style={{height: "100%"}}>
                                <FeatureTab
                                    dashboardContext={this.state.dashboardContext}
                                    selectedFeatureChange={this.setBinIndex}
                                    selectedFeatureIndex={this.state.selectedBinIndex}
                                    featureBins={this.state.featureBins.filter(x => !!x)}
                                    onNext={this.setTab.bind(this, accuracyTabKey)}
                                />
                            </PivotItem>
                            <PivotItem headerText={"Accuracy"} itemKey={accuracyTabKey}>
                                <AccuracyTab
                                    dashboardContext={this.state.dashboardContext}
                                    accuracyPickerProps={accuracyPickerProps}
                                    onNext={this.setTab.bind(this, flights.skipDisparity ? reportTabKey : disparityTabKey)}
                                />
                            </PivotItem>
                            {(flights.skipDisparity === false) && (<PivotItem headerText={"Parity"} itemKey={disparityTabKey}>
                                <ParityTab
                                    dashboardContext={this.state.dashboardContext}
                                    parityPickerProps={parityPickerProps}
                                    onNext={this.setTab.bind(this, reportTabKey)}
                                    onPrevious={this.setTab.bind(this, accuracyTabKey)}
                                />
                            </PivotItem>)}
                        </Pivot>
                    </Stack.Item>}
                {(this.state.activeTabKey === reportTabKey && this.state.selectedModelId !== undefined) &&
                    <WizardReport 
                        dashboardContext={this.state.dashboardContext}
                        metricsCache={this.state.metricCache}
                        selections={this.selections}
                        modelCount={this.props.predictedY.length}
                        accuracyPickerProps={accuracyPickerProps}
                        parityPickerProps={parityPickerProps}
                        featureBinPickerProps={featureBinPickerProps}
                        selectedModelIndex={this.state.selectedModelId}
                        onEditConfigs={this.setTab.bind(this, featureBinTabKey)}
                    />}
                {(this.state.activeTabKey === reportTabKey && this.state.selectedModelId === undefined) &&
                    <ModelComparisonChart
                        dashboardContext={this.state.dashboardContext}
                        metricsCache={this.state.metricCache}
                        selections={this.selections}
                        modelCount={this.props.predictedY.length}
                        accuracyPickerProps={accuracyPickerProps}
                        parityPickerProps={parityPickerProps}
                        featureBinPickerProps={featureBinPickerProps}
                        onEditConfigs={this.setTab.bind(this, featureBinTabKey)}
                    />}
             </Stack>
         );
    }

    private readonly setTab = (key: string) => {
        this.setState({ activeTabKey: key});
    }

    private readonly setSelectedModel = (index: number) => {
        this.setState({selectedModelId: index});
    }

    private readonly setAccuracyKey = (key: string) => {
        const value: Partial<IWizardState> = {selectedAccuracyKey: key};
        if (flights.skipDisparity) {
            value.selectedParityKey = key;
        }
        this.setState(value as IWizardState);
    }

    private readonly setParityKey = (key: string) => {
        this.setState({selectedParityKey: key});
    }

    private readonly setBinIndex = (index: number) => {
        this.binningSet(this.state.featureBins[index])
    }

    private readonly handleTabClick = (item: PivotItem) => {
        this.setState({activeTabKey: item.props.itemKey});
    }

    private readonly binningSet = (value: IBinnedResponse) => {

        if (!value || value.hasError || value.array.length === 0) {
            return;
        }
        const newContext = _.cloneDeep(this.state.dashboardContext);
 
        newContext.binVector = this.generateBinVectorForBin(value, this.state.dashboardContext.dataset);
        newContext.groupNames = this.generateStringLabelsForBins(value, this.state.dashboardContext.modelMetadata);

        this.setState({dashboardContext: newContext, selectedBinIndex: value.featureIndex});
    }

    private generateBinVectorForBin(value: IBinnedResponse, dataset: any[][]): number[] {
        return dataset.map((row, rowIndex) => {
            const featureValue = row[value.featureIndex];
            if (value.rangeType === RangeTypes.categorical) {
                return value.array.indexOf(featureValue);
            } else {
                return value.array.findIndex((group, groupIndex) => { return groupIndex > value.array.length || value.array[groupIndex + 1] > featureValue});
            }
        });
    }

    private generateStringLabelsForBins(bin: IBinnedResponse, modelMetadata: IFairnessModelMetadata): string[] {
        if (bin.rangeType === RangeTypes.categorical) {
            if (bin.array.length === (modelMetadata.featureRanges[bin.featureIndex] as ICategoricalRange).uniqueValues.length) {
                return (bin.array as string[]);
            } else {
                return [].concat(...bin.array, "Other");
            }
        }
        const length = bin.array.length;
        return bin.array.map((val, index) => {
            if (index === length - 1) {
                if (typeof val === "number") {
                    val = val.toFixed(3);
                }
                return `> ${val}`;
            }
            let b = bin.array[index + 1];
            if (typeof val === "number") {
                val = val.toFixed(3);
            }
            if (typeof b === "number") {
                b = b.toFixed(3);
            }
            return `${val} - ${b}`;
        });
    }

    private readonly buildFeatureBins = (metadata: IModelMetadata): IBinnedResponse[] => {
        return metadata.featureNames.map((name, index) => {
            if (metadata.featureIsCategorical[index]) {
                return {
                    hasError: false,
                    array: (metadata.featureRanges[index] as ICategoricalRange).uniqueValues,
                    featureIndex: index,
                    rangeType: RangeTypes.categorical
                };
            }
        });
    }

    private readonly onMetricError = (error: any): void => {
        this.setState({activeTabKey: accuracyTabKey});
    }
}