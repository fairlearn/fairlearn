import React from "react";
import { localization } from "../Localization/localization";
import { Stack, StackItem } from "office-ui-fabric-react/lib/Stack";
import { DataSpecificationBlade } from "./DataSpecificationBlade";
import { Separator } from "office-ui-fabric-react/lib/Separator";
import { IWizardTabProps } from "../IWizardTabProps";
import { WizardFooter } from "./WizardFooter";
import { TileList } from "./TileList";
import { IAccuracyPickerProps } from "../FairnessWizard";

export interface IAccuracyPickingTabProps extends IWizardTabProps {
    accuracyPickerProps: IAccuracyPickerProps
}

export class AccuracyTab extends React.PureComponent<IAccuracyPickingTabProps> {
    private readonly _columnCount = 3;
    render(): React.ReactNode {
        return(
            <Stack horizontal>
                <StackItem grow={2}>
                    <Stack>
                        <h2 style={{fontWeight: "bold"}}>
                            {localization.Accuracy.header}
                        </h2>
                        <p>{localization.Accuracy.body}</p>
                        <StackItem grow={2}>
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
                        <Separator />
                        <WizardFooter onNext={this.props.onNext}/>
                    </Stack>
                </StackItem>
                <DataSpecificationBlade
                        numberRows={this.props.dashboardContext.dataset.length}
                        featureNames={this.props.dashboardContext.modelMetadata.featureNames}/>
            </Stack>
        );
    }
}