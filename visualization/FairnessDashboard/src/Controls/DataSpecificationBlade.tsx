import { Text } from 'office-ui-fabric-react/lib/Text';
import React from 'react';
import { localization } from '../Localization/localization';
import { DataSpecificationBladeStyles } from './DataSpecificationBlade.styles';

export interface IDataSpecProps {
    numberRows: number;
    featureNames: string[];
}

export class DataSpecificationBlade extends React.PureComponent<IDataSpecProps> {
    render(): React.ReactNode {
        const styles = DataSpecificationBladeStyles();
        return (
            <div className={styles.frame}>
                <Text variant={'small'} className={styles.title} block>
                    {localization.dataSpecifications}
                </Text>
                <Text variant={'small'} className={styles.text} block>
                    {this.props.featureNames.length === 1
                        ? localization.singleAttributeCount
                        : localization.formatString(localization.attributesCount, this.props.featureNames.length)}
                </Text>
                <Text variant={'small'} className={styles.text} block>
                    {localization.formatString(localization.instanceCount, this.props.numberRows)}
                </Text>
            </div>
        );
    }
}
