import { initializeIcons } from '@uifabric/icons';
import _ from 'lodash';
import { ICategoricalRange, ModelMetadata, RangeTypes, SelectionContext } from 'mlchartlib';
import { Pivot, PivotItem } from 'office-ui-fabric-react/lib/Pivot';
import { Stack, StackItem } from 'office-ui-fabric-react/lib/Stack';
import { Text } from 'office-ui-fabric-react';
import React from 'react';
import { AccuracyOptions, IAccuracyOption } from './AccuracyMetrics';
import { BinnedResponseBuilder } from './BinnedResponseBuilder';
import { AccuracyTab } from './Controls/AccuracyTab';
import { FeatureTab } from './Controls/FeatureTab';
import { IntroTab } from './Controls/IntroTab';
import { ModelComparisonChart } from './Controls/ModelComparisonChart';
import { ParityTab } from './Controls/ParityTab';
import { IBinnedResponse } from './IBinnedResponse';
import { IFairnessContext, IFairnessModelMetadata } from './IFairnessContext';
import { IFairnessProps, PredictionType, PredictionTypes } from './IFairnessProps';
import { localization } from './Localization/localization';
import { MetricsCache } from './MetricsCache';
import { WizardReport } from './WizardReport';
import { FairnessWizardStyles } from './FairnessWizard.styles';
import { loadTheme } from 'office-ui-fabric-react';
import { defaultTheme } from './Themes';

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

const introTabKey = 'introTab';
const featureBinTabKey = 'featureBinTab';
const accuracyTabKey = 'accuracyTab';
const disparityTabKey = 'disparityTab';
const reportTabKey = 'reportTab';

const flights = {
    skipDisparity: true,
};

export class FairnessWizard extends React.PureComponent<IFairnessProps, IWizardState> {
    private static iconsInitialized = false;

    private static initializeIcons(props: IFairnessProps): void {
        if (FairnessWizard.iconsInitialized === false && props.shouldInitializeIcons !== false) {
            initializeIcons(props.iconUrl);
            FairnessWizard.iconsInitialized = true;
        }
    }

    private static buildModelNames(props: IFairnessProps): string[] {
        return !!props.modelNames && props.modelNames.length === props.predictedY.length
            ? props.modelNames
            : props.predictedY.map((unused, modelIndex) => `Model ${modelIndex}`);
    }

    private static buildInitialFairnessContext(props: IFairnessProps): IFairnessContext {
        return {
            dataset: props.testData,
            trueY: props.trueY,
            predictions: props.predictedY,
            binVector: [],
            groupNames: [],
            modelMetadata: FairnessWizard.buildModelMetadata(props),
            modelNames: FairnessWizard.buildModelNames(props),
        };
    }

    private static buildPrecomputedFairnessContext(props: IFairnessProps): IFairnessContext {
        return {
            dataset: undefined,
            trueY: props.trueY,
            predictions: props.predictedY,
            binVector: props.precomputedFeatureBins[0].binVector,
            groupNames: props.precomputedFeatureBins[0].binLabels,
            modelMetadata: FairnessWizard.buildPrecomputedModelMetadata(props),
            modelNames: FairnessWizard.buildModelNames(props),
        };
    }

    private static getClassLength(props: IFairnessProps): number {
        return _.uniq(props.trueY).length;
    }

    private static buildPrecomputedModelMetadata(props: IFairnessProps): IFairnessModelMetadata {
        let featureNames = props.dataSummary.featureNames;
        if (!featureNames) {
            featureNames = props.precomputedFeatureBins.map((binObject, index) => {
                return binObject.featureBinName || localization.formatString(localization.defaultFeatureNames, index);
            }) as string[];
        }
        const classNames =
            props.dataSummary.classNames ||
            FairnessWizard.buildIndexedNames(FairnessWizard.getClassLength(props), localization.defaultClassNames);
        const featureRanges = props.precomputedFeatureBins.map((binMeta) => {
            return {
                uniqueValues: binMeta.binLabels,
                rangeType: RangeTypes.categorical,
            } as ICategoricalRange;
        });
        return {
            featureNames,
            featureNamesAbridged: featureNames,
            classNames,
            featureIsCategorical: props.precomputedFeatureBins.map((binMeta) => true),
            featureRanges,
            predictionType: props.predictionType,
        };
    }

    private static buildModelMetadata(props: IFairnessProps): IFairnessModelMetadata {
        let featureNames = props.dataSummary.featureNames;
        if (!featureNames) {
            let featureLength = 0;
            if (props.testData && props.testData[0] !== undefined) {
                featureLength = props.testData[0].length;
            }
            featureNames =
                featureLength === 1
                    ? [localization.defaultSingleFeatureName]
                    : FairnessWizard.buildIndexedNames(featureLength, localization.defaultFeatureNames);
        }
        const classNames =
            props.dataSummary.classNames ||
            FairnessWizard.buildIndexedNames(FairnessWizard.getClassLength(props), localization.defaultClassNames);
        const featureIsCategorical = ModelMetadata.buildIsCategorical(
            featureNames.length,
            props.testData,
            props.dataSummary.categoricalMap,
        );
        const featureRanges = ModelMetadata.buildFeatureRanges(
            props.testData,
            featureIsCategorical,
            props.dataSummary.categoricalMap,
        );
        const predictionType = FairnessWizard.determinePredictionType(
            props.trueY,
            props.predictedY,
            props.predictionType,
        );
        return {
            featureNames,
            featureNamesAbridged: featureNames,
            classNames,
            featureIsCategorical,
            featureRanges,
            predictionType,
        };
    }

    private static buildIndexedNames(length: number, baseString: string): string[] {
        return Array.from(Array(length).keys()).map(
            (i) => localization.formatString(baseString, i.toString()) as string,
        );
    }

    private static determinePredictionType(
        trueY: number[],
        predictedYs: number[][],
        specifiedType?: PredictionType,
    ): PredictionType {
        if (
            specifiedType === PredictionTypes.binaryClassification ||
            specifiedType === PredictionTypes.probability ||
            specifiedType === PredictionTypes.regression
        ) {
            return specifiedType;
        }
        const predictedIsPossibleProba = predictedYs.every((predictionVector) =>
            predictionVector.every((x) => x >= 0 && x <= 1),
        );
        const trueIsBinary = _.uniq(trueY).length < 3;
        if (!trueIsBinary) {
            return PredictionTypes.regression;
        }
        if (_.uniq(_.flatten(predictedYs)).length < 3) {
            return PredictionTypes.binaryClassification;
        }
        if (predictedIsPossibleProba) {
            return PredictionTypes.probability;
        }
        return PredictionTypes.regression;
    }

    private selections: SelectionContext;

    constructor(props: IFairnessProps) {
        super(props);
        FairnessWizard.initializeIcons(props);
        if (this.props.locale) {
            localization.setLanguage(this.props.locale);
        }
        let accuracyMetrics: IAccuracyOption[];
        loadTheme(props.theme || defaultTheme);
        this.selections = new SelectionContext('models', 1);
        this.selections.subscribe({
            selectionCallback: (strings: string[]) => {
                const numbers = strings.map((s) => +s);
                this.setSelectedModel(numbers[0]);
            },
        });
        // handle the case of precomputed metrics separately. As it becomes more defined, can integrate with existing code path.
        if (this.props.precomputedMetrics && this.props.precomputedFeatureBins) {
            // we must assume that the same accuracy metrics are provided across models and bins
            accuracyMetrics = this.buildAccuracyListForPrecomputedMetrics();
            const readonlyFeatureBins = this.props.precomputedFeatureBins.map((initialBin, index) => {
                return {
                    hasError: false,
                    array: initialBin.binLabels,
                    labelArray: initialBin.binLabels,
                    featureIndex: index,
                    rangeType: RangeTypes.categorical,
                };
            });
            this.state = {
                accuracyMetrics,
                selectedAccuracyKey: accuracyMetrics[0].key,
                parityMetrics: accuracyMetrics,
                selectedParityKey: accuracyMetrics[0].key,
                dashboardContext: FairnessWizard.buildPrecomputedFairnessContext(props),
                activeTabKey: featureBinTabKey,
                featureBins: readonlyFeatureBins,
                selectedBinIndex: 0,
                selectedModelId: this.props.predictedY.length === 1 ? 0 : undefined,
                metricCache: new MetricsCache(0, 0, undefined, props.precomputedMetrics),
            };
            return;
        }
        const fairnessContext = FairnessWizard.buildInitialFairnessContext(props);

        const featureBins = this.buildFeatureBins(fairnessContext);
        if (featureBins.length > 0) {
            fairnessContext.binVector = this.generateBinVectorForBin(featureBins[0], fairnessContext.dataset);
            fairnessContext.groupNames = featureBins[0].labelArray;
        }

        accuracyMetrics =
            fairnessContext.modelMetadata.predictionType === PredictionTypes.binaryClassification
                ? this.props.supportedBinaryClassificationAccuracyKeys.map((key) => AccuracyOptions[key])
                : fairnessContext.modelMetadata.predictionType === PredictionTypes.regression
                ? this.props.supportedRegressionAccuracyKeys.map((key) => AccuracyOptions[key])
                : this.props.supportedProbabilityAccuracyKeys.map((key) => AccuracyOptions[key]);
        accuracyMetrics = accuracyMetrics.filter((metric) => !!metric);

        this.state = {
            accuracyMetrics,
            selectedAccuracyKey: accuracyMetrics[0].key,
            parityMetrics: accuracyMetrics,
            selectedParityKey: accuracyMetrics[0].key,
            dashboardContext: fairnessContext,
            activeTabKey: introTabKey,
            featureBins,
            selectedBinIndex: 0,
            selectedModelId: this.props.predictedY.length === 1 ? 0 : undefined,
            metricCache: new MetricsCache(featureBins.length, this.props.predictedY.length, this.props.requestMetrics),
        };
    }

    public render(): React.ReactNode {
        const styles = FairnessWizardStyles();
        const accuracyPickerProps = {
            accuracyOptions: this.state.accuracyMetrics,
            selectedAccuracyKey: this.state.selectedAccuracyKey,
            onAccuracyChange: this.setAccuracyKey,
        };
        const parityPickerProps = {
            parityOptions: this.state.parityMetrics,
            selectedParityKey: this.state.selectedParityKey,
            onParityChange: this.setParityKey,
        };
        const featureBinPickerProps = {
            featureBins: this.state.featureBins,
            selectedBinIndex: this.state.selectedBinIndex,
            onBinChange: this.setBinIndex,
        };
        if (this.state.featureBins.length === 0) {
            return (
                <Stack className={styles.frame}>
                    <Stack
                        horizontal
                        horizontalAlign="space-between"
                        verticalAlign="center"
                        className={styles.thinHeader}
                    >
                        <Text variant={'mediumPlus'} className={styles.headerLeft}>
                            {localization.Header.title}
                        </Text>
                    </Stack>
                    <Stack.Item grow={2} className={styles.body}>
                        <Text variant={'mediumPlus'}>{localization.errorOnInputs}</Text>
                    </Stack.Item>
                </Stack>
            );
        }
        return (
            <Stack className={styles.frame}>
                <Stack horizontal horizontalAlign="space-between" verticalAlign="center" className={styles.thinHeader}>
                    <Text variant={'mediumPlus'} className={styles.headerLeft}>
                        {localization.Header.title}
                    </Text>
                </Stack>
                {this.state.activeTabKey === introTabKey && (
                    <StackItem grow={2} className={styles.body}>
                        <IntroTab onNext={this.setTab.bind(this, featureBinTabKey)} />
                    </StackItem>
                )}
                {(this.state.activeTabKey === featureBinTabKey ||
                    this.state.activeTabKey === accuracyTabKey ||
                    this.state.activeTabKey === disparityTabKey) && (
                    <Stack.Item grow={2} className={styles.body}>
                        <Pivot
                            className={styles.pivot}
                            styles={{
                                itemContainer: {
                                    height: '100%',
                                },
                            }}
                            selectedKey={this.state.activeTabKey}
                            onLinkClick={this.handleTabClick}
                        >
                            <PivotItem
                                headerText={localization.Intro.features}
                                itemKey={featureBinTabKey}
                                style={{ height: '100%', paddingLeft: '8px' }}
                            >
                                <FeatureTab
                                    dashboardContext={this.state.dashboardContext}
                                    selectedFeatureChange={this.setBinIndex}
                                    selectedFeatureIndex={this.state.selectedBinIndex}
                                    featureBins={this.state.featureBins.filter((x) => !!x)}
                                    onNext={this.setTab.bind(this, accuracyTabKey)}
                                    saveBin={this.saveBin}
                                />
                            </PivotItem>
                            <PivotItem
                                headerText={localization.accuracyMetric}
                                itemKey={accuracyTabKey}
                                style={{ height: '100%', paddingLeft: '8px' }}
                            >
                                <AccuracyTab
                                    dashboardContext={this.state.dashboardContext}
                                    accuracyPickerProps={accuracyPickerProps}
                                    onNext={this.setTab.bind(
                                        this,
                                        flights.skipDisparity ? reportTabKey : disparityTabKey,
                                    )}
                                    onPrevious={this.setTab.bind(this, featureBinTabKey)}
                                />
                            </PivotItem>
                            {flights.skipDisparity === false && (
                                <PivotItem headerText={'Parity'} itemKey={disparityTabKey}>
                                    <ParityTab
                                        dashboardContext={this.state.dashboardContext}
                                        parityPickerProps={parityPickerProps}
                                        onNext={this.setTab.bind(this, reportTabKey)}
                                        onPrevious={this.setTab.bind(this, accuracyTabKey)}
                                    />
                                </PivotItem>
                            )}
                        </Pivot>
                    </Stack.Item>
                )}
                {this.state.activeTabKey === reportTabKey && this.state.selectedModelId !== undefined && (
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
                    />
                )}
                {this.state.activeTabKey === reportTabKey && this.state.selectedModelId === undefined && (
                    <ModelComparisonChart
                        dashboardContext={this.state.dashboardContext}
                        metricsCache={this.state.metricCache}
                        selections={this.selections}
                        modelCount={this.props.predictedY.length}
                        accuracyPickerProps={accuracyPickerProps}
                        parityPickerProps={parityPickerProps}
                        featureBinPickerProps={featureBinPickerProps}
                        onEditConfigs={this.setTab.bind(this, featureBinTabKey)}
                    />
                )}
            </Stack>
        );
    }

    private readonly buildAccuracyListForPrecomputedMetrics = (): IAccuracyOption[] => {
        const customMetrics: IAccuracyOption[] = [];
        const providedMetrics: IAccuracyOption[] = [];
        Object.keys(this.props.precomputedMetrics[0][0]).forEach((key) => {
            const metric = AccuracyOptions[key];
            if (metric !== undefined) {
                if (metric.userVisible) {
                    providedMetrics.push(metric);
                }
            } else {
                const customIndex = this.props.customMetrics.findIndex((metric) => metric.id === key);
                const customMetric = customIndex !== -1 ? this.props.customMetrics[customIndex] : { id: key };

                customMetrics.push({
                    key,
                    title:
                        customMetric.name ||
                        (localization.formatString(
                            localization.defaultCustomMetricName,
                            customMetrics.length,
                        ) as string),
                    isMinimization: true,
                    isPercentage: true,
                    description: customMetric.description,
                });
            }
        });
        return customMetrics.concat(providedMetrics);
    };

    private readonly setTab = (key: string) => {
        this.setState({ activeTabKey: key });
    };

    private readonly setSelectedModel = (index: number) => {
        this.setState({ selectedModelId: index });
    };

    private readonly setAccuracyKey = (key: string) => {
        const value: Partial<IWizardState> = { selectedAccuracyKey: key };
        if (flights.skipDisparity) {
            value.selectedParityKey = key;
        }
        this.setState(value as IWizardState);
    };

    private readonly setParityKey = (key: string) => {
        this.setState({ selectedParityKey: key });
    };

    private readonly setBinIndex = (index: number) => {
        if (this.props.precomputedMetrics) {
            const newContext = _.cloneDeep(this.state.dashboardContext);

            newContext.binVector = this.props.precomputedFeatureBins[index].binVector;
            newContext.groupNames = this.props.precomputedFeatureBins[index].binLabels;

            this.setState({ dashboardContext: newContext, selectedBinIndex: index });
        } else {
            this.binningSet(this.state.featureBins[index]);
        }
    };

    private readonly handleTabClick = (item: PivotItem) => {
        this.setState({ activeTabKey: item.props.itemKey });
    };

    private readonly binningSet = (value: IBinnedResponse) => {
        if (!value || value.hasError || value.array.length === 0) {
            return;
        }
        const newContext = _.cloneDeep(this.state.dashboardContext);

        newContext.binVector = this.generateBinVectorForBin(value, this.state.dashboardContext.dataset);
        newContext.groupNames = value.labelArray;

        this.setState({ dashboardContext: newContext, selectedBinIndex: value.featureIndex });
    };

    private generateBinVectorForBin(value: IBinnedResponse, dataset: any[][]): number[] {
        return dataset.map((row, rowIndex) => {
            const featureValue = row[value.featureIndex];
            if (value.rangeType === RangeTypes.categorical) {
                // this handles categorical, as well as integers when user requests to treat as categorical
                return value.array.indexOf(featureValue);
            } else {
                return value.array.findIndex((upperLimit, groupIndex) => {
                    return upperLimit >= featureValue;
                });
            }
        });
    }

    private readonly buildFeatureBins = (fairnessContext: IFairnessContext): IBinnedResponse[] => {
        return fairnessContext.modelMetadata.featureNames.map((name, index) => {
            return BinnedResponseBuilder.buildDefaultBin(
                fairnessContext.modelMetadata.featureRanges[index],
                index,
                fairnessContext.dataset,
            );
        });
    };

    private readonly saveBin = (bin: IBinnedResponse): void => {
        this.state.featureBins[bin.featureIndex] = bin;
        this.state.metricCache.clearCache(bin.featureIndex);
        this.binningSet(bin);
    };

    private readonly onMetricError = (error: any): void => {
        this.setState({ activeTabKey: accuracyTabKey });
    };
}
