import { getTheme, IProcessedStyleSet, IStyle, mergeStyleSets, FontWeights, FontSizes } from 'office-ui-fabric-react';

export interface IIntroTabStyles {
    firstSection: IStyle;
    firstSectionTitle: IStyle;
    firstSectionSubtitle: IStyle;
    firstSectionBody: IStyle;
    lowerSection: IStyle;
    stepsContainer: IStyle;
    boldStep: IStyle;
    numericLabel: IStyle;
    explanatoryStep: IStyle;
    explanatoryText: IStyle;
    getStarted: IStyle;
}

export const IntroTabStyles: () => IProcessedStyleSet<IIntroTabStyles> = () => {
    const theme = getTheme();
    return mergeStyleSets<IIntroTabStyles>({
        firstSection: {
            padding: '43px 94px',
            backgroundColor: theme.semanticColors.bodyBackground,
            color: theme.semanticColors.bodyText,
        },
        firstSectionTitle: {
            fontSize: '60px',
            fontWeight: FontWeights.light,
            lineHeight: '82px',
        },
        firstSectionSubtitle: {
            fontSize: '60px',
            fontWeight: FontWeights.semibold,
            lineHeight: '82px',
        },
        firstSectionBody: {
            paddingTop: '30px',
            paddingBottom: '70px',
            maxWidth: '500px',
            fontWeight: FontWeights.semilight,
            lineHeight: '24px',
        },
        lowerSection: {
            padding: '50px 70px 90px 90px',
            backgroundColor: theme.semanticColors.bodyBackground,
            color: theme.semanticColors.bodyText,
            flexGrow: 1,
        },
        stepsContainer: {
            display: 'flex',
            flexDirection: 'row',
            justifyContent: 'space-between',
            borderBottom: '1px solid',
            borderBottomColor: theme.semanticColors.bodyDivider,
            paddingBottom: '38px',
        },
        boldStep: {
            maxWidth: '300px',
            paddingRight: '25px',
            flex: 1,
        },
        numericLabel: {
            fontWeight: FontWeights.bold,
            width: '30px',
            lineHeight: '24px',
        },
        explanatoryStep: {
            maxWidth: '300px',
            paddingRight: '20px',
            flex: 1,
        },
        explanatoryText: {
            paddingTop: '15px',
        },
        getStarted: {
            paddingTop: '30px',
            fontSize: FontSizes.large,
            fontWeight: FontWeights.regular,
        },
    });
};
