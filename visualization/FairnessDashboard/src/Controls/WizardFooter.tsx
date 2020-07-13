import { DefaultButton, PrimaryButton } from 'office-ui-fabric-react/lib/Button';
import React from 'react';
import { localization } from '../Localization/localization';
import { WizardFooterStyles } from './WizardFooter.styles';

export interface IWizardFooterProps {
    onNext: () => void;
    onPrevious?: () => void;
}

export class WizardFooter extends React.PureComponent<IWizardFooterProps> {
    public render(): React.ReactNode {
        const styles = WizardFooterStyles();
        return (
            <div className={styles.frame}>
                <PrimaryButton className={styles.next} text={localization.Footer.next} onClick={this.props.onNext} />
                {!!this.props.onPrevious && (
                    <DefaultButton
                        className={styles.back}
                        text={localization.Footer.back}
                        onClick={this.props.onPrevious}
                    />
                )}
            </div>
        );
    }
}
