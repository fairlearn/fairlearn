import React from "react";
import { localization } from "../Localization/localization";
import { Stack, StackItem } from "office-ui-fabric-react/lib/Stack";
import { DataSpecificationBlade } from "./DataSpecificationBlade";
import { Separator } from "office-ui-fabric-react/lib/Separator";
import { IWizardTabProps } from "../IWizardTabProps";
import { WizardFooter } from "./WizardFooter";
import { TileList, ITileProp } from "./TileList";
import { IParityPickerProps } from "../FairnessWizard";
import { mergeStyleSets } from "@uifabric/styling";
import { ParityTabStyles } from "./ParityTab.styles";

export interface IParityTabProps extends IWizardTabProps {
    parityPickerProps: IParityPickerProps;
}

export class ParityTab extends React.PureComponent<IParityTabProps> {
    render(): React.ReactNode {
        const styles = ParityTabStyles();
        return(
            <Stack horizontal horizontalAlign="space-between" className={styles.frame}>
                <StackItem grow={2}>
                    <Stack className={styles.main}>
                        <h2 className={styles.header}>
                            {localization.Parity.header}
                        </h2>
                        <p className={styles.textBody}>{localization.Parity.body}</p>
                        <StackItem grow={2} className={styles.itemsList}>
                            <TileList
                                items={this.props.parityPickerProps.parityOptions.map((parity, index): ITileProp => {
                                    const selected = this.props.parityPickerProps.selectedParityKey === parity.key;
                                    return {
                                        title: parity.title,
                                        description: parity.description,
                                        onSelect: this.props.parityPickerProps.onParityChange.bind(this, parity.key),
                                        selected
                                    };
                                })}
                            />
                        </StackItem>
                        <WizardFooter onNext={this.props.onNext} onPrevious={this.props.onPrevious}/>
                    </Stack>
                </StackItem>
                <DataSpecificationBlade
                        numberRows={this.props.dashboardContext.trueY.length}
                        featureNames={this.props.dashboardContext.modelMetadata.featureNames}/>
            </Stack>
        );
    }
}