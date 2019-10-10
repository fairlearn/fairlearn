import { IParityPickerProps } from "../FairnessWizard";
import React from "react";
import { localization } from "../Localization/localization";
import { IconButton, DefaultButton } from "office-ui-fabric-react/lib/Button";
import { ComboBox, IComboBox, IComboBoxOption } from "office-ui-fabric-react/lib/ComboBox";
import { IDropdownOption } from "office-ui-fabric-react/lib/Dropdown";
import { Callout } from "office-ui-fabric-react/lib/Callout";

interface IState {
    showCallout: boolean;
}

export class ParityPicker extends React.PureComponent <IParityPickerProps, IState> {
    private _parityDropdownHelpId: string = "_parityDropdownHelpId";
    constructor(props: IParityPickerProps) {
        super(props);
        this.state = { showCallout: false};
    }
    render(): React.ReactNode {
        const options: IDropdownOption[] = this.props.parityOptions.map(option => {
            return {
                key: option.key,
                text: option.title
            };
        });

        return (<div>
            <div className="selector">
                <div className="selector-label">
                    <span>{"TODO"}</span>
                    <IconButton
                        id={this._parityDropdownHelpId}
                        iconProps={{ iconName: 'Info' }}
                        title={"TODO"}
                        ariaLabel="Info"
                        onClick={this.onOpen}
                        styles={{ root: { marginBottom: -3, color: 'rgb(0, 120, 212)' } }}
                    />
                </div>
                <ComboBox
                    className="path-selector"
                    selectedKey={this.props.selectedParityKey}
                    onChange={this.onParityChange}
                    options={options}
                    ariaLabel={"Parity selector"}
                    useComboBoxAsMenuWidth={true}
                /> 
            </div>
            {this.state.showCallout && (
                <Callout
                    target={'#' + this._parityDropdownHelpId}
                    setInitialFocus={true}
                    onDismiss={this.onDismiss}
                    role="alertdialog">
                    <div className="callout-info">
                        <DefaultButton onClick={this.onDismiss}>{localization.close}</DefaultButton>
                    </div>
                </Callout>
                )}
        </div>);
    }

    private readonly onDismiss = (): void => {
        this.setState({ showCallout: false});
    }

    private readonly onOpen = (): void => {
        this.setState({ showCallout: true});
    }

    private readonly onParityChange = (event: React.FormEvent<IComboBox>, item: IComboBoxOption) => {
        this.props.onParityChange(item.key as string);
    }
}