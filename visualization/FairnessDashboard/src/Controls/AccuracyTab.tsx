import React from "react";
import { localization } from "../Localization/localization";
import { Stack, StackItem } from "office-ui-fabric-react/lib/Stack";
import { DataSpecificationBlade } from "./DataSpecificationBlade";
import { IWizardTabProps } from "../IWizardTabProps";
import { WizardFooter } from "./WizardFooter";
import { TileList } from "./TileList";
import { IAccuracyPickerProps } from "../FairnessWizard";
import { mergeStyleSets } from "@uifabric/styling";
import { PredictionTypes } from "../IFairnessProps";

export interface IAccuracyPickingTabProps extends IWizardTabProps {
    accuracyPickerProps: IAccuracyPickerProps
}

export class AccuracyTab extends React.PureComponent<IAccuracyPickingTabProps> {
    private static readonly classNames = mergeStyleSets({
        itemCell: [
          {
            padding: "30px 36px 20px 0",
            width: "100%",
            position: "relative",
            float: "left",
            cursor: "pointer",
            boxSizing: "border-box",
            borderBottom: "1px solid #CCCCCC",
            selectors: {
              '&:hover': { background: "lightgray" }
            }
          }
        ],
        iconClass: {
            fontSize: "20px",
            position: "absolute",
            right: "10px",
            top: "10px"
        },
        itemsList: {
            overflowY: "auto"
        },
        frame: {
            height: "100%",
        },
        main: {
            height: "100%",
            maxWidth: "750px",
            flex: 1
        },
        header: {
            color: "#333333",
            fontSize: "32px",
            lineHeight: "39px",
            fontWeight: "100"
        },
        textBody: {
            paddingTop: "12px",
            fontSize: "18px",
            lineHeight: "24px",
            fontWeight: "300"
        }
    });
    render(): React.ReactNode {
        return(
            <Stack horizontal horizontalAlign="space-between" className={AccuracyTab.classNames.frame}>
                <Stack className={AccuracyTab.classNames.main}>
                    <h2 className={AccuracyTab.classNames.header}>
                        {localization.Accuracy.header}
                    </h2>
                    {/* <p className={AccuracyTab.classNames.textBody}>{localization.formatString(localization.Accuracy.body,
                        this.props.dashboardContext.modelMetadata.predictionType === PredictionTypes.binaryClassification ?
                            localization.Accuracy.binaryClassifier :
                            localization.Accuracy.regressor
                        )}</p> */}
                    <StackItem grow={2} className={AccuracyTab.classNames.itemsList}>
                        <TileList
                            items={this.props.accuracyPickerProps.accuracyOptions.map((accuracy, index) => {
                                return {
                                    title: accuracy.title,
                                    description: accuracy.description,
                                    onSelect: this.props.accuracyPickerProps.onAccuracyChange.bind(this, accuracy.key),
                                    selected: this.props.accuracyPickerProps.selectedAccuracyKey === accuracy.key
                                };
                            })}
                        />
                    </StackItem>
                    <WizardFooter onNext={this.props.onNext} onPrevious={this.props.onPrevious}/>
                </Stack>
                <DataSpecificationBlade
                        numberRows={this.props.dashboardContext.dataset.length}
                        featureNames={this.props.dashboardContext.modelMetadata.featureNames}/>
            </Stack>
        );
    }
}