import { getTheme, IProcessedStyleSet, IStyle, mergeStyleSets } from 'office-ui-fabric-react';

export interface IWizardFooterStyles {
    frame: IStyle;
    next: IStyle;
    back: IStyle;
}

export const WizardFooterStyles: () => IProcessedStyleSet<IWizardFooterStyles> = () => {
    const theme = getTheme();
    return mergeStyleSets<IWizardFooterStyles>({
        frame: {
            display: 'inline-flex',
            flexDirection: 'row-reverse',
            paddingTop: '10px',
            paddingBottom: '10px',
        },
        next: {
            height: '44px',
            padding: '12px',
            fontSize: '18px',
            lineHeight: '24px',
            fontWeight: '400',
            marginLeft: '10px',
        },
        back: {
            height: '44px',
            padding: '12px',
            fontSize: '18px',
            lineHeight: '24px',
            fontWeight: '400',
        },
    });
};
